// Versão CUDA do Barenn: mesmo treino em mini-batch (soma dos gradientes) do
// barenn.cpp, com forward, backward e atualização na GPU. Reaproveita a leitura
// de dados, a perda e a avaliação do barenn.cpp (incluído abaixo).
#define BARENN_SEM_MAIN
#include "barenn.cpp"

#include <cuda_runtime.h>

#define CUDA_OK(chamada)                                                        \
    do {                                                                        \
        cudaError_t e_ = (chamada);                                             \
        if (e_ != cudaSuccess) {                                                \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(e_) +     \
                                " em " + __FILE__ + ":" + to_string(__LINE__)); \
        }                                                                       \
    } while (0)

namespace {

constexpr int threads = 256;

// Uma thread por (amostra, neurônio): produto escalar na mesma ordem da CPU.
__global__ void forward(const float* in, int in_f, const float* w, const float* b,
                        int out_f, bool relu, float* z, float* a, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * out_f) return;
    int s = k / out_f, i = k % out_f;
    float sum = b[i];
    for (int j = 0; j < in_f; ++j) sum += in[s * in_f + j] * w[i * in_f + j];
    z[k] = sum;
    a[k] = relu ? fmaxf(0.0f, sum) : sum;
}

// Perda quadrática por amostra e o erro (derivada em relação ao escore).
__global__ void perda_erro(const float* escores, const int* rotulos, int n,
                           float* erro, float* perda_amostra) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    float total = 0;
    for (int i = 0; i < Barenn::classes; ++i) {
        float e = escores[s * Barenn::classes + i] - (i == rotulos[s] ? 1.0f : 0.0f);
        total += 0.5f * e * e;
        erro[s * Barenn::classes + i] = e;
    }
    perda_amostra[s] = total;
}

// delta = grad_out * (z > 0), a derivada da ReLU.
__global__ void delta_relu(const float* grad_out, const float* z, float* delta, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) delta[k] = grad_out[k] * (z[k] > 0.0f ? 1.0f : 0.0f);
}

// grad_in[s][j] = soma_i delta[s][i] * w[i][j], com os pesos antes da atualização.
__global__ void grad_entrada(const float* delta, const float* w, int out_f, int in_f,
                             float* grad_in, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * in_f) return;
    int s = k / in_f, j = k % in_f;
    float sum = 0;
    for (int i = 0; i < out_f; ++i) sum += delta[s * out_f + i] * w[i * in_f + j];
    grad_in[k] = sum;
}

// Uma thread por peso: soma o gradiente do lote e aplica o SGD. A thread j == 0
// de cada linha também atualiza o viés.
__global__ void atualizar(const float* delta, const float* in, int out_f, int in_f,
                          float* w, float* b, int n, float taxa) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= out_f * in_f) return;
    int i = k / in_f, j = k % in_f;
    float g = 0;
    for (int s = 0; s < n; ++s) g += delta[s * out_f + i] * in[s * in_f + j];
    w[k] -= taxa * g;
    if (j == 0) {
        float gb = 0;
        for (int s = 0; s < n; ++s) gb += delta[s * out_f + i];
        b[i] -= taxa * gb;
    }
}

int blocos(int n) { return (n + threads - 1) / threads; }

// Camada na GPU: pesos e viés copiados do objeto da CPU.
struct CamadaGpu {
    int in_f, out_f;
    float *w, *b;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste [lote]]\n";
        return 1;
    }
    const string arquivo_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc >= 3 ? argv[2] : "optdigits.tes";

    try {
        int lote = 32;
        if (argc == 4 && (!(istringstream(argv[3]) >> lote) || lote < 1)) {
            throw runtime_error("O lote deve ser um inteiro maior ou igual a 1.");
        }
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        Barenn::DenseLayer* camadas_cpu[3] = {&l1, &l2, &l3};
        const int tam[4] = {Barenn::entradas, 32, 16, Barenn::classes};
        const size_t n_treino = treino.size();

        // Dados em formato contíguo (host).
        vector<float> h_x(n_treino * Barenn::entradas);
        vector<int> h_y(n_treino);
        for (size_t s = 0; s < n_treino; ++s) {
            copy(treino[s].imagem.begin(), treino[s].imagem.end(),
                 h_x.begin() + s * Barenn::entradas);
            h_y[s] = treino[s].rotulo;
        }
        vector<float> h_perda(static_cast<size_t>(epocas) * n_treino);

        // O driver inicializa o contexto CUDA na primeira chamada (~190 ms). Isso é
        // um custo único do programa, não do treino, então fica fora do cronômetro.
        CUDA_OK(cudaFree(nullptr));

        // A medição inclui as cópias Host->Device e Device->Host, a alocação e o
        // treino de todas as épocas, até a GPU terminar.
        const auto inicio = chrono::steady_clock::now();

        float *d_x, *d_perda;
        int* d_y;
        CUDA_OK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_y, h_y.size() * sizeof(int)));
        CUDA_OK(cudaMalloc(&d_perda, h_perda.size() * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(int), cudaMemcpyHostToDevice));

        CamadaGpu gpu[3];
        for (int l = 0; l < 3; ++l) {
            gpu[l] = {tam[l], tam[l + 1], nullptr, nullptr};
            CUDA_OK(cudaMalloc(&gpu[l].w, sizeof(float) * tam[l] * tam[l + 1]));
            CUDA_OK(cudaMalloc(&gpu[l].b, sizeof(float) * tam[l + 1]));
            vector<float> w;
            for (auto& linha : camadas_cpu[l]->Pesos()) w.insert(w.end(), linha.begin(), linha.end());
            CUDA_OK(cudaMemcpy(gpu[l].w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(gpu[l].b, camadas_cpu[l]->Vies().data(),
                               sizeof(float) * tam[l + 1], cudaMemcpyHostToDevice));
        }

        // Buffers de um lote: somas z, ativações a, deltas e gradientes da entrada.
        float *z[3], *a[3], *delta[2], *gin[2], *erro;
        for (int l = 0; l < 3; ++l) {
            CUDA_OK(cudaMalloc(&z[l], sizeof(float) * lote * tam[l + 1]));
            CUDA_OK(cudaMalloc(&a[l], sizeof(float) * lote * tam[l + 1]));
        }
        for (int l = 0; l < 2; ++l) {
            CUDA_OK(cudaMalloc(&delta[l], sizeof(float) * lote * tam[l + 1]));
            CUDA_OK(cudaMalloc(&gin[l], sizeof(float) * lote * tam[l + 1]));
        }
        CUDA_OK(cudaMalloc(&erro, sizeof(float) * lote * Barenn::classes));

        for (int epoca = 0; epoca < epocas; ++epoca) {
            for (size_t ini = 0; ini < n_treino; ini += lote) {
                const int n = static_cast<int>(min<size_t>(lote, n_treino - ini));
                const float* x = d_x + ini * Barenn::entradas;
                const float* entrada[3] = {x, a[0], a[1]};
                for (int l = 0; l < 3; ++l) {
                    forward<<<blocos(n * tam[l + 1]), threads>>>(
                        entrada[l], tam[l], gpu[l].w, gpu[l].b, tam[l + 1], l < 2,
                        z[l], a[l], n);
                }
                perda_erro<<<blocos(n), threads>>>(
                    a[2], d_y + ini, n, erro, d_perda + static_cast<size_t>(epoca) * n_treino + ini);

                // Camada 3: o delta é o próprio erro (saída linear).
                grad_entrada<<<blocos(n * tam[2]), threads>>>(erro, gpu[2].w, tam[3], tam[2], gin[1], n);
                atualizar<<<blocos(tam[3] * tam[2]), threads>>>(erro, a[1], tam[3], tam[2], gpu[2].w, gpu[2].b, n, taxa);
                // Camada 2.
                delta_relu<<<blocos(n * tam[2]), threads>>>(gin[1], z[1], delta[1], n * tam[2]);
                grad_entrada<<<blocos(n * tam[1]), threads>>>(delta[1], gpu[1].w, tam[2], tam[1], gin[0], n);
                atualizar<<<blocos(tam[2] * tam[1]), threads>>>(delta[1], a[0], tam[2], tam[1], gpu[1].w, gpu[1].b, n, taxa);
                // Camada 1 (não precisa de grad_in).
                delta_relu<<<blocos(n * tam[1]), threads>>>(gin[0], z[0], delta[0], n * tam[1]);
                atualizar<<<blocos(tam[1] * tam[0]), threads>>>(delta[0], x, tam[1], tam[0], gpu[0].w, gpu[0].b, n, taxa);
            }
        }
        CUDA_OK(cudaGetLastError());

        // Pesos e perdas de volta para a CPU (a avaliação roda na CPU).
        CUDA_OK(cudaMemcpy(h_perda.data(), d_perda, h_perda.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int l = 0; l < 3; ++l) {
            vector<float> w(static_cast<size_t>(tam[l]) * tam[l + 1]);
            CUDA_OK(cudaMemcpy(w.data(), gpu[l].w, w.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < tam[l + 1]; ++i) {
                copy(w.begin() + static_cast<size_t>(i) * tam[l], w.begin() + static_cast<size_t>(i + 1) * tam[l],
                     camadas_cpu[l]->Pesos()[i].begin());
            }
            CUDA_OK(cudaMemcpy(camadas_cpu[l]->Vies().data(), gpu[l].b,
                               sizeof(float) * tam[l + 1], cudaMemcpyDeviceToHost));
        }
        CUDA_OK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        for (int l = 0; l < 3; ++l) { cudaFree(gpu[l].w); cudaFree(gpu[l].b); cudaFree(z[l]); cudaFree(a[l]); }
        for (int l = 0; l < 2; ++l) { cudaFree(delta[l]); cudaFree(gin[l]); }
        cudaFree(erro); cudaFree(d_x); cudaFree(d_y); cudaFree(d_perda);

        array<double, epocas> perdas{};
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double total = 0;
            for (size_t s = 0; s < n_treino; ++s) total += h_perda[static_cast<size_t>(epoca) * n_treino + s];
            perdas[epoca] = total / n_treino;
        }
        cout << fixed << setprecision(4) << "TREINO | " << n_treino << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
