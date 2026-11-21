#!/bin/sh
# Roda o binário N vezes e resume o "Tempo de treino" (mediana, min, max, desvio).
# Uso: ./bench.sh [N] [binario] [args...]   (padrão: 10 ./barenn)
n=${1:-10}
bin=${2:-./barenn}
[ $# -gt 2 ] && shift 2 || shift $#
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //') ($(nproc) threads)"
echo "g++: $(g++ --version | head -1) | $(grep '^CXXFLAGS' Makefile)"
for i in $(seq "$n"); do "$bin" "$@" | sed -n 's/^Tempo de treino \(.*\) s$/\1/p'; done |
    sort -n | awk '{t[NR]=$1; s+=$1; q+=$1*$1}
        END {m=s/NR; sd=sqrt(q/NR-m*m); med=(NR%2)?t[(NR+1)/2]:(t[NR/2]+t[NR/2+1])/2;
             printf "N=%d | mediana %.4f s | min %.4f | max %.4f | desvio %.4f\n", NR, med, t[1], t[NR], sd}'
