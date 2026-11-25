# Run the embeddings on Kaggle's GPU (all 13 classes, ~1 h)

1. Open the dataset page on Kaggle and click **New Notebook**. The dataset is attached automatically.
2. In the right panel, under **Settings**:
   - **Accelerator** → **GPU T4 x2**
   - **Internet** → **On** (needed to download the model weights)
3. **Cell 1**:
   ```
   !pip install -q open_clip_torch
   ```
4. **Cell 2**: type `%%writefile embed.py` on the first line, then paste the whole of `deep/embed.py` below it.
5. **Cell 3**:
   ```
   !python embed.py --data-root /kaggle/input --out /kaggle/working/emb_all.npz --batch-size 64 --workers 4
   ```
   It should print `images=42499`. If it shows a different number, stop and tell me.
6. Click **Save Version** → **Save & Run All**. When it finishes, download **`emb_all.npz`** (~400 MB) from the **Output** tab. Ignore the `emb_all_chunks` folder.
7. Put the file at `CV/outputs/deep/embeddings/emb_all.npz`, then run:
   ```
   /opt/anaconda3/bin/python deep/probe.py --task 13
   /opt/anaconda3/bin/python deep/evaluate.py --run probe_13
   ```
