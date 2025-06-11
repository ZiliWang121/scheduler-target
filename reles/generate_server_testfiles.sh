#!/usr/bin/env bash
FILES=("64kb.dat" "2mb.dat" "8mb.dat" "64mb.dat")
for f in "${FILES[@]}"; do
  [[ -f $f ]] || {
    echo "生成 $f …"
    if [[ $f == *kb.dat ]]; then
      size=${f%%kb.dat}
      dd if=/dev/urandom of="$f" bs=1024 count=$size status=none
    else
      size=${f%%mb.dat}
      dd if=/dev/urandom of="$f" bs=1M count=$size status=none
    fi
  }
done
ls -lh *.dat
