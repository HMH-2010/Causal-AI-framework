from training import predict_scores
from model import predict_survival


def predict_potential_survival(
    model,
    X,
    event_times,
    baseline_survival,
    time_points,
    device,
):
    scores = predict_scores(
        model,
        X,
        device,
    )

    survival = predict_survival(
        scores,
        event_times,
        baseline_survival,
        time_points,
    )

    return scores, survival


def compute_ite(
    survival_surgery,
    survival_neoadjuvant,
):
    return survival_neoadjuvant - survival_surgery
