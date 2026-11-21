#!/bin/sh
# Compara a saída do binário com a referência (golden), ignorando o tempo.
# Uso: ./check.sh [binario] [golden] [argumentos do binario...]
#      (padrão: ./barenn golden.txt, sem argumentos)
bin=${1:-./barenn}
golden=${2:-golden.txt}
[ $# -gt 2 ] && shift 2 || shift $#
if "$bin" "$@" | grep -v '^Tempo de treino' | diff - "$golden" >/dev/null; then
    echo "OK: saída idêntica a $golden"
else
    echo "FALHA: saída difere de $golden" >&2
    "$bin" "$@" | grep -v '^Tempo de treino' | diff - "$golden" | head -20 >&2
    exit 1
fi
