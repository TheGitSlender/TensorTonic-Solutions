import numpy as np
from math import ceil

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    n = len(production_log)

    prod_preds = np.array([x["prediction"] for x in production_log])
    prod_actuals = np.array([x["actual"] for x in production_log])
    shadow_preds = np.array([x["prediction"] for x in shadow_log])
    shadow_actuals = np.array([x["actual"] for x in shadow_log])
    shadow_latencies = np.array([x["latency_ms"] for x in shadow_log])

    prod_acc = np.sum(prod_preds == prod_actuals)
    shadow_acc = np.sum(shadow_preds == shadow_actuals)
    agreement_rate = np.mean(prod_preds == shadow_preds)
    shadow_latency_p95 = shadow_latencies[ceil(.95*n)-1]
    accuracy_gain = (shadow_acc/n) - (prod_acc/n)

    promote_value = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )
    return {
        "promote": promote_value,
        "metrics": {
            "shadow_accuracy": shadow_acc/n,
            "production_accuracy": prod_acc/n,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate
        }
    }