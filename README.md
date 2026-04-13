# burn_dragon

Focused `burn_dragon` workspace for peer-to-peer language training.

Included crates:
- `burn_dragon_p2p`
- `burn_dragon_language`
- `burn_dragon_train`
- `burn_dragon_core`
- `burn_dragon_kernel`
- `burn_dragon_universality`
- `burn_dragon_tokenizer`
- `burn_dragon_checkpoint`
- `xtask`

Retained model and kernel surface:
- linear attention
- Mamba3 state-space duality
- dense/fused recurrent attention and projection kernels needed by the language training line
- the NCA and ClimbMix configs and deploy assets used by `burn_dragon_p2p`

Checked-in experiment/deploy assets live under:
- `crates/burn_dragon_p2p/deploy/profiles/`
- `crates/burn_dragon_p2p/deploy/terraform/aws/`
- `xtask/src/main.rs`

This repo intentionally omits the unrelated vision, multimodal, CLI, gameplay, legacy sequence-family, and browser-unrelated monorepo work. Generated run artifacts, dataset blobs, and Terraform provider caches are ignored.

Deployment workflows:
- `.github/workflows/deploy-burn-dragon-p2p-aws.yml`: manual AWS bootstrap/edge deployment
- `.github/workflows/deploy-burn-dragon-p2p-pages.yml`: separate manual GitHub Pages deployment for the browser peer shell
  - enable GitHub Pages in repository settings and set the source to `GitHub Actions`
