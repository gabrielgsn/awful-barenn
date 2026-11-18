// Versão CUDA do Barenn: mesmo treino em mini-batch (soma dos gradientes) do
// barenn.cpp, na GPU. Reaproveita a leitura de dados, a perda e a avaliação do
// barenn.cpp (incluído abaixo).
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

// Rede 64 -> 32 -> 16 -> 10. Um lote cabe em um único bloco: uma thread por
// (amostra, neurônio da camada 1), no máximo 32 amostras.
constexpr int N0 = Barenn::entradas, N1 = 32, N2 = 16, N3 = Barenn::classes;
constexpr int max_lote = 32;
constexpr int threads = max_lote * N1;

// Pesos na memória compartilhada, transpostos e com uma coluna de preenchimento:
// w[j * (saida + 1) + i]. Ao ler com i variando entre as threads de um warp (forward)
// ou com j variando (gradiente da entrada), os acessos caem em bancos diferentes.
__device__ __forceinline__ int pos(int j, int i, int saida) { return j * (saida + 1) + i; }

// Copia pesos [saida][entrada] da memória global para a compartilhada (e de volta).
__device__ void carregar(float* s, const float* g, int entrada, int saida) {
    for (int k = threadIdx.x; k < entrada * saida; k += threads) {
        s[pos(k % entrada, k / entrada, saida)] = g[k];
    }
}
__device__ void guardar(float* g, const float* s, int entrada, int saida) {
    for (int k = threadIdx.x; k < entrada * saida; k += threads) {
        g[k] = s[pos(k % entrada, k / entrada, saida)];
    }
}

// Soma o gradiente das n amostras e aplica o SGD a uma camada (uma thread por peso).
// A thread com j == 0 de cada neurônio também atualiza o viés.
__device__ void atualizar(float* w, float* b, const float* delta, const float* in,
                          int entrada, int saida, int n, float taxa) {
    for (int k = threadIdx.x; k < entrada * saida; k += threads) {
        int j = k / saida, i = k % saida;
        float g = 0;
        for (int s = 0; s < n; ++s) g += delta[s * saida + i] * in[s * entrada + j];
        w[pos(j, i, saida)] -= taxa * g;
        if (j == 0) {
            float gb = 0;
            for (int s = 0; s < n; ++s) gb += delta[s * saida + i];
            b[i] -= taxa * gb;
        }
    }
}

// Estado do treino na memória compartilhada: pesos, vieses e os buffers de um lote.
struct Estado {
    float w1[N0 * (N1 + 1)], w2[N1 * (N2 + 1)], w3[N2 * (N3 + 1)];
    float b1[N1], b2[N2], b3[N3];
    float x[max_lote * N0];
    float a1[max_lote * N1], a2[max_lote * N2], a3[max_lote * N3];
    float d3[max_lote * N3], d2[max_lote * N2], d1[max_lote * N1];
};

// Um lote de n amostras (e.x já tem as imagens; y aponta para os rótulos do lote):
// forward, perda, backward e atualização, com __syncthreads() entre as etapas. Todas
// as threads do bloco chamam esta função. A soma de cada neurônio segue a ordem da
// CPU (j crescente, a partir do viés); o gradiente soma as amostras em ordem crescente.
__device__ void lote(Estado& e, const int* y, int n, float taxa, float* perda_amostra) {
    const int t = threadIdx.x;
    if (t < n * N1) {
        int s = t / N1, i = t % N1;
        float sum = e.b1[i];
        for (int j = 0; j < N0; ++j) sum += e.x[s * N0 + j] * e.w1[pos(j, i, N1)];
        e.a1[t] = fmaxf(0.0f, sum);
    }
    __syncthreads();
    if (t < n * N2) {
        int s = t / N2, i = t % N2;
        float sum = e.b2[i];
        for (int j = 0; j < N1; ++j) sum += e.a1[s * N1 + j] * e.w2[pos(j, i, N2)];
        e.a2[t] = fmaxf(0.0f, sum);
    }
    __syncthreads();
    if (t < n * N3) {
        int s = t / N3, i = t % N3;
        float sum = e.b3[i];
        for (int j = 0; j < N2; ++j) sum += e.a2[s * N2 + j] * e.w3[pos(j, i, N3)];
        // Saída linear. O erro é a derivada da perda em relação ao escore.
        e.d3[t] = sum - (i == y[s] ? 1.0f : 0.0f);
    }
    __syncthreads();
    if (t < n) {
        float total = 0;
        for (int i = 0; i < N3; ++i) total += 0.5f * e.d3[t * N3 + i] * e.d3[t * N3 + i];
        perda_amostra[t] = total;
    }

    // Camada 3: gradiente da entrada (com os pesos antigos) e atualização. A ReLU da
    // camada 2 passa o gradiente onde a ativação é positiva (a > 0 equivale a z > 0).
    if (t < n * N2) {
        int s = t / N2, j = t % N2;
        float sum = 0;
        for (int i = 0; i < N3; ++i) sum += e.d3[s * N3 + i] * e.w3[pos(j, i, N3)];
        e.d2[t] = sum * (e.a2[t] > 0.0f ? 1.0f : 0.0f);
    }
    __syncthreads();
    atualizar(e.w3, e.b3, e.d3, e.a2, N2, N3, n, taxa);
    // Camada 2.
    if (t < n * N1) {
        int s = t / N1, j = t % N1;
        float sum = 0;
        for (int i = 0; i < N2; ++i) sum += e.d2[s * N2 + i] * e.w2[pos(j, i, N2)];
        e.d1[t] = sum * (e.a1[t] > 0.0f ? 1.0f : 0.0f);
    }
    __syncthreads();
    atualizar(e.w2, e.b2, e.d2, e.a1, N1, N2, n, taxa);
    __syncthreads();
    // Camada 1 (não precisa do gradiente da entrada).
    atualizar(e.w1, e.b1, e.d1, e.x, N0, N1, n, taxa);
    __syncthreads();
}

// Kernel persistente: um único lançamento treina todas as épocas e todos os lotes.
// Os pesos ficam na memória compartilhada do começo ao fim (carregados e guardados
// uma vez). O x do próximo lote é lido da memória global para registradores no
// começo do lote atual (2 float por thread, o lote tem no máximo 2048), para a
// latência da leitura se sobrepor ao cálculo.
__global__ void treinar(float* w1, float* b1, float* w2, float* b2, float* w3,
                        float* b3, const float* x, const int* y, int n_treino,
                        int tam_lote, int epocas, float taxa, float* perda) {
    static_assert(threads * 2 == max_lote * N0, "2 float por thread cobrem um lote");
    __shared__ Estado e;
    const int t = threadIdx.x;
    carregar(e.w1, w1, N0, N1);
    carregar(e.w2, w2, N1, N2);
    carregar(e.w3, w3, N2, N3);
    for (int k = t; k < N1; k += threads) e.b1[k] = b1[k];
    for (int k = t; k < N2; k += threads) e.b2[k] = b2[k];
    for (int k = t; k < N3; k += threads) e.b3[k] = b3[k];

    const int lotes = (n_treino + tam_lote - 1) / tam_lote;
    const int total = epocas * lotes;
    // Prefetch do primeiro lote.
    float p0 = 0, p1 = 0;
    {
        const int n = min(tam_lote, n_treino);
        if (t < n * N0) p0 = x[t];
        if (t + threads < n * N0) p1 = x[t + threads];
    }
    for (int g = 0; g < total; ++g) {
        const int epoca = g / lotes, ini = (g % lotes) * tam_lote;
        const int n = min(tam_lote, n_treino - ini);
        // Os registradores do prefetch viram o x do lote atual.
        if (t < n * N0) e.x[t] = p0;
        if (t + threads < n * N0) e.x[t + threads] = p1;
        __syncthreads();
        // Dispara a leitura do lote seguinte, que se sobrepõe ao cálculo abaixo.
        if (g + 1 < total) {
            const int ini_p = ((g + 1) % lotes) * tam_lote;
            const int n_p = min(tam_lote, n_treino - ini_p);
            if (t < n_p * N0) p0 = x[ini_p * N0 + t];
            if (t + threads < n_p * N0) p1 = x[ini_p * N0 + t + threads];
        }
        lote(e, y + ini, n, taxa, perda + static_cast<size_t>(epoca) * n_treino + ini);
    }
    guardar(w1, e.w1, N0, N1);
    guardar(w2, e.w2, N1, N2);
    guardar(w3, e.w3, N2, N3);
    for (int k = threadIdx.x; k < N1; k += threads) b1[k] = e.b1[k];
    for (int k = threadIdx.x; k < N2; k += threads) b2[k] = e.b2[k];
    for (int k = threadIdx.x; k < N3; k += threads) b3[k] = e.b3[k];
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste [lote]]\n";
        return 1;
    }
    const string arquivo_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc >= 3 ? argv[2] : "optdigits.tes";

    try {
        int lote_tam = 32;
        if (argc == 4 && (!(istringstream(argv[3]) >> lote_tam) || lote_tam < 1 ||
                          lote_tam > max_lote)) {
            throw runtime_error("O lote deve ser um inteiro entre 1 e " +
                                to_string(max_lote) + " na versao CUDA.");
        }
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        Barenn::DenseLayer l1(N0, N1);
        Barenn::DenseLayer l2(N1, N2);
        Barenn::DenseLayer l3(N2, N3, true);
        Barenn::DenseLayer* camadas_cpu[3] = {&l1, &l2, &l3};
        const int tam[4] = {N0, N1, N2, N3};
        const size_t n_treino = treino.size();

        // Dados em formato contíguo (host).
        vector<float> h_x(n_treino * N0);
        vector<int> h_y(n_treino);
        for (size_t s = 0; s < n_treino; ++s) {
            copy(treino[s].imagem.begin(), treino[s].imagem.end(), h_x.begin() + s * N0);
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

        float *dw[3], *db[3];
        for (int l = 0; l < 3; ++l) {
            CUDA_OK(cudaMalloc(&dw[l], sizeof(float) * tam[l] * tam[l + 1]));
            CUDA_OK(cudaMalloc(&db[l], sizeof(float) * tam[l + 1]));
            vector<float> w;
            for (auto& linha : camadas_cpu[l]->Pesos()) w.insert(w.end(), linha.begin(), linha.end());
            CUDA_OK(cudaMemcpy(dw[l], w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(db[l], camadas_cpu[l]->Vies().data(),
                               sizeof(float) * tam[l + 1], cudaMemcpyHostToDevice));
        }

        // Um único lançamento para o treino inteiro.
        treinar<<<1, threads>>>(dw[0], db[0], dw[1], db[1], dw[2], db[2], d_x, d_y,
                                static_cast<int>(n_treino), lote_tam, epocas, taxa, d_perda);
        CUDA_OK(cudaGetLastError());

        // Pesos e perdas de volta para a CPU (a avaliação roda na CPU).
        CUDA_OK(cudaMemcpy(h_perda.data(), d_perda, h_perda.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int l = 0; l < 3; ++l) {
            vector<float> w(static_cast<size_t>(tam[l]) * tam[l + 1]);
            CUDA_OK(cudaMemcpy(w.data(), dw[l], w.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < tam[l + 1]; ++i) {
                copy(w.begin() + static_cast<size_t>(i) * tam[l], w.begin() + static_cast<size_t>(i + 1) * tam[l],
                     camadas_cpu[l]->Pesos()[i].begin());
            }
            CUDA_OK(cudaMemcpy(camadas_cpu[l]->Vies().data(), db[l],
                               sizeof(float) * tam[l + 1], cudaMemcpyDeviceToHost));
        }
        CUDA_OK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        for (int l = 0; l < 3; ++l) { cudaFree(dw[l]); cudaFree(db[l]); }
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_perda);

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
