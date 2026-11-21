#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Gradientes de uma camada, somados ao longo das amostras de um bloco.
struct Grad {
    vector<vector<float>> w;
    vector<float> b;

    void zerar() {
        for (auto& linha : w) {
            fill(linha.begin(), linha.end(), 0.0f);
        }
        fill(b.begin(), b.end(), 0.0f);
    }
};

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // weights[i][j] liga a entrada j ao neurônio de saída i.
    vector<vector<float>> weights;
    vector<float> bias;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size, vector<float>(input_size)), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (auto& row : weights) {
            for (float& weight : row) {
                weight = distribution(gen);
            }
        }
    }

    Grad NovoGrad() const {
        return {vector<vector<float>>(out_features, vector<float>(in_features)),
                vector<float>(out_features)};
    }

    // z (soma antes da ativação) e out pertencem ao chamador: a camada não guarda
    // estado de nenhuma amostra, então várias threads podem usá-la ao mesmo tempo.
    void Forward(const vector<float>& input, vector<float>& z,
                 vector<float>& out) const {
        z.resize(out_features);
        out.resize(out_features);
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        auto ativar = [&](int i, float sum) {
            z[i] = sum;
            out[i] = final ? sum : max(0.0f, sum);
        };
        // Cada neurônio soma o viés aos produtos entre entradas e pesos. Quatro
        // neurônios por vez: as quatro somas são independentes e se sobrepõem na
        // CPU (ILP). A ordem de soma de cada neurônio não muda.
        int i = 0;
        for (; i + 4 <= out_features; i += 4) {
            const float* w0 = weights[i].data();
            const float* w1 = weights[i + 1].data();
            const float* w2 = weights[i + 2].data();
            const float* w3 = weights[i + 3].data();
            float s0 = bias[i], s1 = bias[i + 1], s2 = bias[i + 2], s3 = bias[i + 3];
            for (int j = 0; j < in_features; ++j) {
                s0 += input[j] * w0[j];
                s1 += input[j] * w1[j];
                s2 += input[j] * w2[j];
                s3 += input[j] * w3[j];
            }
            ativar(i, s0);
            ativar(i + 1, s1);
            ativar(i + 2, s2);
            ativar(i + 3, s3);
        }
        for (; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i][j];
            }
            ativar(i, sum);
        }
    }

    // Soma em g os gradientes dos pesos e do viés desta amostra. Os pesos só mudam
    // em Aplicar. grad_in recebe o gradiente da entrada; a primeira camada passa
    // nullptr, porque não há camada antes dela.
    void Backward(const vector<float>& input, const vector<float>& z,
                  const vector<float>& grad_out, vector<float>& delta,
                  vector<float>* grad_in, Grad& g) const {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        delta.resize(out_features);
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // Um passe por linha de pesos (contíguo). O gradiente do peso é delta
        // vezes a entrada; o do viés é o próprio delta.
        // d é copiado para um escalar: lendo delta[i] dentro do laço, o compilador
        // teria de supor que as escritas em gi/gw podem alterá-lo.
        const float* x = input.data();
        if (grad_in) {
            grad_in->assign(in_features, 0.0f);
            float* gi = grad_in->data();
            for (int i = 0; i < out_features; ++i) {
                const float d = delta[i];
                const float* w = weights[i].data();
                float* gw = g.w[i].data();
                for (int j = 0; j < in_features; ++j) {
                    gi[j] += d * w[j];
                    gw[j] += d * x[j];
                }
                g.b[i] += d;
            }
        } else {
            for (int i = 0; i < out_features; ++i) {
                const float d = delta[i];
                float* gw = g.w[i].data();
                for (int j = 0; j < in_features; ++j) {
                    gw[j] += d * x[j];
                }
                g.b[i] += d;
            }
        }
    }

    // SGD: w -= taxa * (soma dos gradientes dos n blocos, sempre na ordem 0..n-1).
    // A soma é feita no gradiente do bloco 0, que é zerado antes do próximo lote.
    template <class F>
    void Aplicar(int n, float taxa, F grad_do_bloco) {
        Grad& soma = grad_do_bloco(0);
        for (int b = 1; b < n; ++b) {
            const Grad& gb = grad_do_bloco(b);
            for (int i = 0; i < out_features; ++i) {
                float* s = soma.w[i].data();
                const float* x = gb.w[i].data();
                for (int j = 0; j < in_features; ++j) {
                    s[j] += x[j];
                }
                soma.b[i] += gb.b[i];
            }
        }
        for (int i = 0; i < out_features; ++i) {
            float* w = weights[i].data();
            const float* s = soma.w[i].data();
            for (int j = 0; j < in_features; ++j) {
                w[j] -= taxa * s[j];
            }
            bias[i] -= taxa * soma.b[i];
        }
    }
};

// Buffers de trabalho de um bloco de amostras (um conjunto por bloco, nunca
// compartilhado entre threads).
struct Trabalho {
    array<vector<float>, 3> z, a, delta, grad_in;
    array<Grad, 3> g;
    vector<float> erro;
    double perda = 0;
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    Trabalho t;
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        l1.Forward(amostra.imagem, t.z[0], t.a[0]);
        l2.Forward(t.a[0], t.z[1], t.a[1]);
        l3.Forward(t.a[1], t.z[2], t.a[2]);
        const auto& escores = t.a[2];
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste [lote]]\n";
        return 1;
    }
    const string arquivo_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc >= 3 ? argv[2] : "optdigits.tes";

    try {
        // Amostras entre duas atualizações dos pesos. O padrão, 1, é o SGD por
        // amostra do código original.
        int lote = 1;
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
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Os pesos só são atualizados a cada lote de amostras (lote 1 = SGD por
        // amostra, como no código original). Cada lote se divide em blocos de
        // `bloco` amostras; cada bloco acumula o seu gradiente e os blocos são
        // somados sempre na mesma ordem, então o resultado não depende de quem
        // calcula cada bloco.
        constexpr int bloco = 4;
        const int max_blocos = (lote + bloco - 1) / bloco;
        vector<Barenn::Trabalho> trab(max_blocos);
        for (auto& t : trab) {
            t.g = {l1.NovoGrad(), l2.NovoGrad(), l3.NovoGrad()};
        }

        // Forward, perda e backward (acumulando gradientes) das amostras [ini, fim).
        auto processar = [&](Barenn::Trabalho& t, size_t ini, size_t fim) {
            for (auto& g : t.g) {
                g.zerar();
            }
            t.perda = 0;
            t.erro.resize(Barenn::classes);
            for (size_t s = ini; s < fim; ++s) {
                const auto& amostra = treino[s];
                l1.Forward(amostra.imagem, t.z[0], t.a[0]);
                l2.Forward(t.a[0], t.z[1], t.a[1]);
                l3.Forward(t.a[1], t.z[2], t.a[2]);
                t.perda += Barenn::perda(t.a[2], amostra.rotulo, &t.erro);

                // Propaga o gradiente da última camada até a primeira.
                l3.Backward(t.a[1], t.z[2], t.erro, t.delta[2], &t.grad_in[2], t.g[2]);
                l2.Backward(t.a[0], t.z[1], t.grad_in[2], t.delta[1], &t.grad_in[1], t.g[1]);
                l1.Backward(amostra.imagem, t.z[0], t.grad_in[1], t.delta[0], nullptr, t.g[0]);
            }
        };

        // A medição inclui forward, perda, backward e atualização de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t ini = 0; ini < treino.size(); ini += lote) {
                const size_t fim = min(ini + lote, treino.size());
                const int n_blocos = static_cast<int>((fim - ini + bloco - 1) / bloco);
                for (int b = 0; b < n_blocos; ++b) {
                    const size_t b_ini = ini + static_cast<size_t>(b) * bloco;
                    processar(trab[b], b_ini, min(b_ini + bloco, fim));
                }
                for (int b = 0; b < n_blocos; ++b) {
                    perda_total += trab[b].perda;
                }
                // Soma (não média) dos gradientes do lote: cada amostra pesa como no
                // SGD por amostra. Com lote 64 e taxa 0,02 o treino diverge.
                const float passo = taxa;
                l1.Aplicar(n_blocos, passo, [&](int b) -> Barenn::Grad& { return trab[b].g[0]; });
                l2.Aplicar(n_blocos, passo, [&](int b) -> Barenn::Grad& { return trab[b].g[1]; });
                l3.Aplicar(n_blocos, passo, [&](int b) -> Barenn::Grad& { return trab[b].g[2]; });
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
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
