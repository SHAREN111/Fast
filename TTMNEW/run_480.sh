bash base_480.sh raw 0 raw
bash base_480.sh raw 1 raw
bash base_480.sh raw 2 raw
bash base_480.sh raw 3 raw
bash base_480.sh raw 4 raw

bash base_480.sh TTM1 0 ttm
bash base_480.sh TTM1 1 ttm
bash base_480.sh TTM1 2 ttm
bash base_480.sh TTM1 3 ttm
bash base_480.sh TTM1 4 ttm

bash base_480.sh FastVAR1 0 fastvar
bash base_480.sh FastVAR1 1 fastvar
bash base_480.sh FastVAR1 2 fastvar
bash base_480.sh FastVAR1 3 fastvar
bash base_480.sh FastVAR1 4 fastvar

bash base_480.sh FastVAR2 0 fastvar
bash base_480.sh FastVAR2 1 fastvar
bash base_480.sh FastVAR2 2 fastvar
bash base_480.sh FastVAR2 3 fastvar
bash base_480.sh FastVAR2 4 fastvar

bash base_480.sh FastVAR3 0 fastvar
bash base_480.sh FastVAR3 1 fastvar
bash base_480.sh FastVAR3 2 fastvar
bash base_480.sh FastVAR3 3 fastvar
bash base_480.sh FastVAR3 4 fastvar

bash base_480.sh FastVAR4 0 fastvar
bash base_480.sh FastVAR4 1 fastvar
bash base_480.sh FastVAR4 2 fastvar
bash base_480.sh FastVAR4 3 fastvar
bash base_480.sh FastVAR4 4 fastvar

bash base_480.sh SparseVAR_0.80 0 sparsevar
bash base_480.sh SparseVAR_0.80 1 sparsevar
bash base_480.sh SparseVAR_0.80 2 sparsevar
bash base_480.sh SparseVAR_0.80 3 sparsevar
bash base_480.sh SparseVAR_0.80 4 sparsevar

bash base_480.sh SparseVAR_0.87 0 sparsevar
bash base_480.sh SparseVAR_0.87 1 sparsevar
bash base_480.sh SparseVAR_0.87 2 sparsevar
bash base_480.sh SparseVAR_0.87 3 sparsevar
bash base_480.sh SparseVAR_0.87 4 sparsevar

bash base_480.sh SparseVAR_0.85 0 sparsevar
bash base_480.sh SparseVAR_0.85 1 sparsevar
bash base_480.sh SparseVAR_0.85 2 sparsevar
bash base_480.sh SparseVAR_0.85 3 sparsevar
bash base_480.sh SparseVAR_0.85 4 sparsevar

bash base_480.sh SparseVAR_0.90 0 sparsevar
bash base_480.sh SparseVAR_0.90 1 sparsevar
bash base_480.sh SparseVAR_0.90 2 sparsevar
bash base_480.sh SparseVAR_0.90 3 sparsevar
bash base_480.sh SparseVAR_0.90 4 sparsevar

wait


python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/gpu.py


