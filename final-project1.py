import re, string, tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np, pandas as pd

# ─────────────────────────────────────────────
# 1. PREPROCESSOR
# ─────────────────────────────────────────────
class Preprocessor:
    STOP = {"the","is","at","on","and","a","to","in","of","for","it","this",
            "that","be","as","are","was","with","by","from","or","but","not",
            "you","your","i","me","my","we","our","he","she","they","them",
            "his","her","its","have","has","had","do","did","so","if","then",
            "there","their","will","would","can","could","should"}

    def __init__(self): self.vocab = {}

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)
        text = re.sub(r"\d+", " num ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [t for t in text.split() if t not in self.STOP and len(t) > 2]

    def build_vocab(self, docs, min_df=3, max_features=1500):
        df_counter = {}
        for d in docs:
            for t in set(self.clean(d)):
                df_counter[t] = df_counter.get(t, 0) + 1
        kept = sorted([(w, c) for w, c in df_counter.items() if c >= min_df],
                      key=lambda x: -x[1])[:max_features]
        self.vocab = {w: i for i, (w, _) in enumerate(kept)}
        return self.vocab

    def vectorize(self, text):
        v = np.zeros(len(self.vocab))
        for t in self.clean(text):
            if t in self.vocab: v[self.vocab[t]] += 1
        return v

# ─────────────────────────────────────────────
# 2. NAIVE BAYES
# ─────────────────────────────────────────────
class NaiveBayes:
    def __init__(self, alpha=1.0): self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior, self.log_like = {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.log_prior[c] = np.log(len(Xc) / len(X))
            counts = Xc.sum(0) + self.alpha
            self.log_like[c] = np.log(counts / counts.sum())
        return self

    def _scores(self, X):
        S = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            S[:, i] = self.log_prior[c] + X @ self.log_like[c]
        return S

    def predict(self, X): return self.classes[np.argmax(self._scores(X), 1)]

    def predict_proba(self, X):
        S = self._scores(X); S -= S.max(1, keepdims=True)
        e = np.exp(S); return e / e.sum(1, keepdims=True)

# ─────────────────────────────────────────────
# 3. FAST DECISION TREE
# ─────────────────────────────────────────────
class Node:
    __slots__ = ("feat","left","right","value")
    def __init__(self): self.feat = self.left = self.right = self.value = None

class DecisionTree:
    def __init__(self, max_depth=8, min_samples=5):
        self.max_depth, self.min_samples = max_depth, min_samples
        self.root = None

    @staticmethod
    def _gini(y):
        if len(y) == 0: return 0.0
        p = np.mean(y); return 1 - p*p - (1-p)*(1-p)

    def _best_split(self, X, y):
        Xb = (X > 0).astype(np.float32)
        n = len(y); p = np.mean(y); base = 1 - p*p - (1-p)*(1-p)
        if base == 0: return None
        n1 = Xb.sum(0); n0 = n - n1
        s1 = (Xb.T @ y).astype(np.float32); s0 = y.sum() - s1
        with np.errstate(divide="ignore", invalid="ignore"):
            g1 = 1 - (s1/n1)**2 - ((n1-s1)/n1)**2
            g0 = 1 - (s0/n0)**2 - ((n0-s0)/n0)**2
        g1[np.isnan(g1)] = 0; g0[np.isnan(g0)] = 0
        gains = base - (n1*g1 + n0*g0) / n
        gains[(n1 < self.min_samples) | (n0 < self.min_samples)] = -1
        f = int(np.argmax(gains))
        return f if gains[f] > 0 else None

    def _build(self, X, y, depth):
        n = Node(); n.value = int(round(np.mean(y))) if len(y) else 0
        if depth >= self.max_depth or len(y) < 2*self.min_samples or self._gini(y) == 0:
            return n
        f = self._best_split(X, y)
        if f is None: return n
        mask = X[:, f] > 0
        if mask.sum() < self.min_samples or (~mask).sum() < self.min_samples: return n
        n.feat = f
        n.left  = self._build(X[~mask], y[~mask], depth+1)
        n.right = self._build(X[ mask], y[ mask], depth+1)
        return n

    def fit(self, X, y): self.root = self._build(X, y, 0); return self

    def _walk(self, x):
        node = self.root
        while node.feat is not None:
            node = node.right if x[node.feat] > 0 else node.left
        return node.value

    def predict(self, X): return np.array([self._walk(x) for x in X])

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1-p, p])

# ─────────────────────────────────────────────
# 4. DATA LOADER + EVALUATOR
# ─────────────────────────────────────────────
class DataLoader:
    LABELS = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    def load(self, path):
        for enc in ("latin-1","utf-8","cp1252"):
            try: df = pd.read_csv(path, encoding=enc); break
            except UnicodeDecodeError: continue
        cols = {c.lower().strip(): c for c in df.columns}
        lc = next((cols[k] for k in ("label","v1","class","category") if k in cols), None)
        mc = next((cols[k] for k in ("message","v2","text","email","body") if k in cols), None)
        if not lc or not mc: raise ValueError("CSV needs label+v1 or message+v2 columns.")
        out = pd.DataFrame({"label": df[lc].astype(str).str.lower().str.strip(),
                            "message": df[mc].astype(str)})
        out["label"] = out["label"].map(self.LABELS)
        out = out.dropna().astype({"label": int})
        return out[out["message"].str.len() > 0].reset_index(drop=True)

    def split(self, df, test_size=0.2, seed=42):
        idx = np.random.default_rng(seed).permutation(len(df))
        cut = int(len(df)*(1-test_size))
        return df.iloc[idx[:cut]], df.iloc[idx[cut:]]

def evaluate(y_true, y_pred):
    tp = int(np.sum((y_true==1)&(y_pred==1))); tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1))); fn = int(np.sum((y_true==1)&(y_pred==0)))
    acc = (tp+tn)/len(y_true); prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,
            "tp":tp,"tn":tn,"fp":fp,"fn":fn}

# ─────────────────────────────────────────────
# 5. TKINTER APP
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("Spam Detector | NB vs DT")
        root.geometry("900x760")
        self.pre  = Preprocessor()
        self.nb   = NaiveBayes()
        self.dt   = DecisionTree(max_depth=8, min_samples=5)
        self.m_nb, self.m_dt, self.trained = {}, {}, False
        self._ui()

    def _ui(self):
        tk.Label(self.root, text="📧 Spam Detection System",
                 font=("Segoe UI", 18, "bold"), fg="#1f4e79").pack(pady=(12,2))
        tk.Label(self.root, text="Naive Bayes + Decision Tree — Combined Verdict",
                 font=("Segoe UI", 10, "italic"), fg="#666").pack(pady=(0,10))

        top = ttk.LabelFrame(self.root, text=" 1. Train Both Models "); top.pack(fill="x", padx=14, pady=6)
        self.ds_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(top, textvariable=self.ds_var, foreground="#555").pack(anchor="w", padx=10, pady=(6,0))
        row = ttk.Frame(top); row.pack(fill="x", padx=10, pady=8)
        ttk.Button(row, text="📂 Load & Train", command=self.train).pack(side="left")
        ttk.Button(row, text="📊 Compare", command=self.show_metrics).pack(side="left", padx=8)
        ttk.Button(row, text="🗑 Clear", command=self.clear).pack(side="left")

        mid = ttk.LabelFrame(self.root, text=" 2. Classify a Message "); mid.pack(fill="both", expand=True, padx=14, pady=6)
        ttk.Label(mid, text="Paste an email or SMS:").pack(anchor="w", padx=10, pady=(6,0))
        self.text = scrolledtext.ScrolledText(mid, height=6, wrap="word", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=6)
        ttk.Button(mid, text="🔍 Classify Message", command=self.classify).pack(pady=(0,8))

        # single combined result line
        self.result = tk.Label(self.root, text="", font=("Segoe UI", 14, "bold"))
        self.result.pack(pady=4)
        self.detail = tk.Label(self.root, text="", font=("Segoe UI", 10), fg="#555")
        self.detail.pack()

        bot = ttk.LabelFrame(self.root, text=" 3. Model Comparison "); bot.pack(fill="both", expand=True, padx=14, pady=(6,12))
        self.mbox = scrolledtext.ScrolledText(bot, height=14, state="disabled",
                                              font=("Consolas", 10), bg="#f7f7f7")
        self.mbox.pack(fill="both", expand=True, padx=10, pady=8)

    def train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            self.root.config(cursor="watch"); self.root.update()
            df = DataLoader().load(path)
            tr, te = DataLoader().split(df)
            self.pre.build_vocab(tr["message"], min_df=3, max_features=1500)
            Xtr = np.array([self.pre.vectorize(m) for m in tr["message"]])
            Xte = np.array([self.pre.vectorize(m) for m in te["message"]])
            ytr, yte = tr["label"].values, te["label"].values

            self.nb.fit(Xtr, ytr); self.dt.fit(Xtr, ytr)
            self.m_nb = evaluate(yte, self.nb.predict(Xte))
            self.m_dt = evaluate(yte, self.dt.predict(Xte))
            self.trained = True

            self.ds_var.set(f"Loaded: {len(df)} | Train: {len(tr)} | Test: {len(te)} | Vocab: {len(self.pre.vocab)}")
            self.show_metrics()
            messagebox.showinfo("Done", f"NB acc: {self.m_nb['accuracy']:.4f}\nDT acc: {self.m_dt['accuracy']:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.root.config(cursor="")

    # ── COMBINED VERDICT ─────────────────────
    def classify(self):
        if not self.trained:
            return messagebox.showwarning("Not trained", "Train models first.")
        txt = self.text.get("1.0", tk.END).strip()
        if not txt: return messagebox.showinfo("Empty", "Enter a message.")

        x = self.pre.vectorize(txt).reshape(1, -1)
        nb_p = int(self.nb.predict(x)[0])
        dt_p = int(self.dt.predict(x)[0])
        nb_s = self.nb.predict_proba(x)[0][list(self.nb.classes).index(1)] if 1 in self.nb.classes else 0
        dt_s = self.dt.predict_proba(x)[0][1]

        # Combined decision: both agree → that; disagree → use avg probability
        if nb_p == dt_p:
            final = nb_p
        else:
            final = int(((nb_s + dt_s) / 2) >= 0.5)

        # Build ONE verdict line
        if final == 1:
            self.result.config(text="🚫  SPAM", fg="#b00020")
        else:
            self.result.config(text="✅  HAM (Not Spam)", fg="#0b8043")

        # Short supporting line
        nb_txt = "SPAM" if nb_p == 1 else "HAM"
        dt_txt = "SPAM" if dt_p == 1 else "HAM"
        agree  = "agree" if nb_p == dt_p else "disagree → tie-break by probability"
        self.detail.config(
            text=f"NB: {nb_txt} ({nb_s*100:.1f}%)  •  DT: {dt_txt} ({dt_s*100:.1f}%)  •  {agree}")

    def show_metrics(self):
        if not self.m_nb: return
        a, b = self.m_nb, self.m_dt
        def row(n, x, y): return f"  {n:<10} {x:>8.4f}  {y:>8.4f}   {'◀' if x>y else ('▶' if y>x else '=')}"
        nb_w = sum(1 for k in ("accuracy","precision","recall","f1") if a[k] > b[k])
        dt_w = sum(1 for k in ("accuracy","precision","recall","f1") if b[k] > a[k])
        win  = "NAIVE BAYES" if nb_w >= dt_w else "DECISION TREE"
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║          MODEL COMPARISON — TEST SET                     ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Metric      NaiveBayes  DecisionTree   Winner           ║",
            "╠══════════════════════════════════════════════════════════╣",
            row("Accuracy",  a["accuracy"],  b["accuracy"]),
            row("Precision", a["precision"], b["precision"]),
            row("Recall",    a["recall"],    b["recall"]),
            row("F1",        a["f1"],        b["f1"]),
            "╠══════════════════════════════════════════════════════════╣",
            f"║  TP (spam→spam)  NB:{a['tp']:<4}   DT:{b['tp']:<4}                    ║",
            f"║  TN (ham→ham)    NB:{a['tn']:<4}   DT:{b['tn']:<4}                    ║",
            f"║  FP (ham→spam)   NB:{a['fp']:<4}   DT:{b['fp']:<4}                    ║",
            f"║  FN (spam→ham)   NB:{a['fn']:<4}   DT:{b['fn']:<4}                    ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Overall winner: {win:<15}(NB:{nb_w} | DT:{dt_w})           ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        self.mbox.config(state="normal"); self.mbox.delete("1.0", tk.END)
        self.mbox.insert(tk.END, "\n".join(lines)); self.mbox.config(state="disabled")

    def clear(self):
        self.text.delete("1.0", tk.END)
        self.result.config(text=""); self.detail.config(text="")


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()