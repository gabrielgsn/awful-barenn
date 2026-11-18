#!/bin/sh
# Compara a saída de um binário com um golden por TOLERÂNCIA (para a versão CUDA,
# cuja ordem de soma em float difere da CPU). Ignora a linha de tempo.
# Passa se: acurácia difere até 0,5 ponto percentual e cada perda por época difere
# até 10% (relativo). A tolerância de perda é larga porque diferenças de
# arredondamento se acumulam ao longo de 6 mil atualizações. A acurácia, que é o
# que importa, tem tolerância estreita.
# Uso: ./compara.sh binario golden [argumentos do binario...]
bin=$1; golden=$2; shift 2
"$bin" "$@" | grep -v '^Tempo de treino' > /tmp/compara_saida.txt || exit 1
awk '
    NR == FNR { if ($1 == "Epoca") ref[FNR] = $6; if ($1 == "Acuracia") { a = $2; sub("%", "", a); ref_ac = a } next }
    $1 == "Epoca" { n++; d = ($6 - ref[FNR]) / ref[FNR]; if (d < 0) d = -d; if (d > 0.10) { print "perda fora da tolerancia:", $0, "(ref " ref[FNR] ")"; erro = 1 } }
    $1 == "Acuracia" { a = $2; sub("%", "", a); d = a - ref_ac; if (d < 0) d = -d; if (d > 0.5) { print "acuracia fora da tolerancia:", a, "(ref " ref_ac ")"; erro = 1 } ac = 1 }
    END { if (erro || !ac || n == 0) { print "FALHA"; exit 1 } print "OK: dentro da tolerancia de " ARGV[1] }
' "$golden" /tmp/compara_saida.txt
