"""
Step 2.4: Train EEGNet across all 9 BCI IV-2a subjects and summarize results.
"""
import numpy as np
from train import train_subject


def main():
    results = []
    for subject_id in range(1, 10):
        result = train_subject(subject_id=subject_id, n_epochs=100, verbose=True)
        results.append(result)

    accs = [r["best_val_acc"] for r in results]

    print(f"\n{'='*60}")
    print("MULTI-SUBJECT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Subject':<10}{'Val Acc':<10}{'Train N':<10}{'Test N':<10}")
    for r in results:
        print(f"{r['subject_id']:<10}{r['best_val_acc']:<10.3f}{r['n_train']:<10}{r['n_test']:<10}")

    print(f"\nMean accuracy across subjects: {np.mean(accs):.3f}")
    print(f"Std deviation across subjects: {np.std(accs):.3f}")
    print(f"Best subject: {results[np.argmax(accs)]['subject_id']} ({max(accs):.3f})")
    print(f"Worst subject: {results[np.argmin(accs)]['subject_id']} ({min(accs):.3f})")


if __name__ == "__main__":
    main()
