import torch as th
import numpy as np
from layer import ADC_GCN
from trainer import PrepareData # We'll reuse your data loader
import time
import sys

def run_stress_test():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # 1. Load the real graph and metadata
    print(">>> Initializing AURELION EDR Environment...")
    class Args:
        dataset = 'malware'
        nhid = 200
        dropout = 0.5
    
    data = PrepareData(Args())
    test_ids = data.test_lst
    
    # 2. Load the trained model
    model = ADC_GCN(nfeat=data.nfeat_dim, nhid=200, nclass=2, dropout=0.5)
    model.load_state_dict(th.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    print(f"\n[!] STRESS TEST STARTING: {len(test_ids)} Unseen Samples Detected.")
    print("-" * 50)
    
    results = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    
    with th.no_grad():
        # Run inference on the whole graph
        logits = model(data.features.to(device), data.adj.to(device))
        probs = th.softmax(logits, dim=1)
        predictions = th.argmax(probs, dim=1)

        # Iterate through the first 20 for the "Live" demo feel
        for i, node_id in enumerate(test_ids[:20]):
            true_label = data.target[node_id]
            pred_label = predictions[node_id].item()
            confidence = probs[node_id][pred_label].item()
            
            # Label strings
            status = "MALICIOUS" if pred_label == 1 else "BENIGN"
            actual = "MALWARE" if true_label == 1 else "SAFE"
            
            # Visual feedback
            marker = "✔" if pred_label == true_label else "✘"
            color = "\033[91m" if status == "MALICIOUS" else "\033[92m"
            reset = "\033[0m"
            
            print(f"Sample #{node_id:05d} | {color}{status:10s}{reset} | Conf: {confidence:.4f} | Truth: {actual:8s} | {marker}")
            time.sleep(0.1) # Simulate real-time processing

        # Calculate Full Stats for all test samples
        for node_id in test_ids:
            true_l = data.target[node_id]
            pred_l = predictions[node_id].item()
            
            if true_l == 1 and pred_l == 1: results["TP"] += 1
            elif true_l == 0 and pred_l == 1: results["FP"] += 1
            elif true_l == 0 and pred_l == 0: results["TN"] += 1
            elif true_l == 1 and pred_l == 0: results["FN"] += 1

    # 3. Final Report Card
    print("-" * 50)
    total = len(test_ids)
    accuracy = (results["TP"] + results["TN"]) / total
    recall = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
    
    print(f"\n--- AURELION EDR FINAL REPORT ---")
    print(f"Total Samples Scanned: {total}")
    print(f"Ransomware Detected:   {results['TP']}")
    print(f"False Positives:       {results['FP']}")
    print(f"Undetected Threats:    {results['FN']}")
    print(f"\n>>> FINAL ACCURACY: {accuracy*100:.2f}%")
    print(f">>> RECALL (PROTECTION RATE): {recall*100:.2f}%")
    print("-" * 35)

if __name__ == "__main__":
    run_stress_test()