import json
from pathlib import Path


def main() -> None:
    profile_path = Path("crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json")
    profile = json.loads(profile_path.read_text())
    browser = profile["browser"]
    train_source = browser["train_source"]
    eval_source = browser["eval_source"]
    policy = browser["capability_policy"]

    assert browser["max_train_batches"] <= 4, browser["max_train_batches"]
    assert browser["max_eval_batches"] <= 1, browser["max_eval_batches"]
    assert (
        policy["browser_wgpu_memory_budget_bytes"] == 6 * 1024 * 1024 * 1024
    ), policy["browser_wgpu_memory_budget_bytes"]
    assert train_source["type"] == "generated_nca", train_source["type"]
    assert eval_source["type"] == "generated_nca", eval_source["type"]
    assert train_source["max_documents"] <= 16, train_source["max_documents"]
    assert eval_source["max_documents"] <= 4, eval_source["max_documents"]

    print("browser-profile-budget-ok")


if __name__ == "__main__":
    main()
