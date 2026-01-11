bash base_480.sh TTM3 0 ttm
bash base_480.sh TTM3 1 ttm
bash base_480.sh TTM3 2 ttm
bash base_480.sh TTM3 3 ttm
bash base_480.sh TTM3 4 ttm

bash base_480.sh TTM4 0 ttm
bash base_480.sh TTM4 1 ttm
bash base_480.sh TTM4 2 ttm
bash base_480.sh TTM4 3 ttm
bash base_480.sh TTM4 4 ttm

bash base_480.sh TTM2 0 ttm
bash base_480.sh TTM2 1 ttm
bash base_480.sh TTM2 2 ttm
bash base_480.sh TTM2 3 ttm
bash base_480.sh TTM2 4 ttm

bash base_480.sh FastVAR5 0 fastvar
bash base_480.sh FastVAR5 1 fastvar
bash base_480.sh FastVAR5 2 fastvar
bash base_480.sh FastVAR5 3 fastvar
bash base_480.sh FastVAR5 4 fastvar

bash base_480.sh FastVAR6 0 fastvar
bash base_480.sh FastVAR6 1 fastvar
bash base_480.sh FastVAR6 2 fastvar
bash base_480.sh FastVAR6 3 fastvar
bash base_480.sh FastVAR6 4 fastvar

bash base_480.sh SparseVAR_0.96 0 sparsevar
bash base_480.sh SparseVAR_0.96 1 sparsevar
bash base_480.sh SparseVAR_0.96 2 sparsevar
bash base_480.sh SparseVAR_0.96 3 sparsevar
bash base_480.sh SparseVAR_0.96 4 sparsevar

bash base_480.sh SparseVAR_0.92 0 sparsevar
bash base_480.sh SparseVAR_0.92 1 sparsevar
bash base_480.sh SparseVAR_0.92 2 sparsevar
bash base_480.sh SparseVAR_0.92 3 sparsevar
bash base_480.sh SparseVAR_0.92 4 sparsevar

bash base_480.sh SparseVAR_0.94 0 sparsevar
bash base_480.sh SparseVAR_0.94 1 sparsevar
bash base_480.sh SparseVAR_0.94 2 sparsevar
bash base_480.sh SparseVAR_0.94 3 sparsevar
bash base_480.sh SparseVAR_0.94 4 sparsevar

wait


python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/gpu.py


