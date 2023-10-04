nsys profile --stats=true -o souffle-swin_trans_O0 -f true  python3 swin_transformer_naive_main.py | tee souffle-swin_trans_O0.log
sqlite3 --csv souffle-swin_trans_O0.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
