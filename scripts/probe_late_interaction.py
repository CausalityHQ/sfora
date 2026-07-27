"""Does LATE INTERACTION beat global pooling for deep metric learning?

The DML benchmark tradition (CUB/Cars/SOP/In-Shop) represents each image as ONE
pooled vector. Document retrieval abandoned that years ago: ColBERT (Khattab &
Zaharia, SIGIR 2020) keeps a SET of token vectors and scores by MaxSim -- for each
query token take its best-matching document token, then sum. ColPali (2024) carried
it to document images. Landmark retrieval has related machinery (DELF; Tolias et al.
selective match kernels). Nobody appears to have applied it to the global-embedding
DML benchmark family.

Why it should matter HERE specifically, tied to this repo's own measurements: 5-8% of
CUB test images are in NOBODY's top-10 (antihubs) and they fail their own retrieval by
21 points. Global average pooling averages away the single locally-discriminative
region -- a wing-bar, a headlight shape -- that would separate two fine-grained
classes. That is exactly the image that becomes an antihub under a single-vector
metric.

RESULT (Cars, cars_pa_teacher.pt, 3x3=9 tokens, same weights both ways):
    global pooled           R@1 = 0.8306
    MaxSim late interaction R@1 = 0.8159   (-1.47 pt)

So late interaction does NOT come for free from a globally-pooled model. That is the
expected-but-worth-knowing outcome: the model was TRAINED with average+max pooling, so
nothing ever asked its individual spatial descriptors to be discriminative on their
own. ColBERT trains WITH late interaction; this only shows the read-out cannot be
swapped in after the fact. It does not settle whether training with MaxSim would work
- that needs a real training run and a new loss path.

This tests the read-out with NO retraining. A trained ResNet-50 DML model is
    backbone -> avgpool -> fc(2048 -> 512)
so applying `fc` at each spatial location instead of pooling first yields local
descriptors for free. Same weights, same training, only the read-out changes.
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from sfora.data import load_image_retrieval_examples  # noqa: E402
from sfora.image_end_to_end import ImageEndToEndConfig, _default_transform_factory  # noqa: E402

CKPT = "reports/emb/cars_pa_teacher.pt"
DATASET = "cars"
GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def build():
    import torchvision

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 512)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model.eval()


@torch.no_grad()
def embed(model, examples, transform, device, grid):
    """Return (global 512-d, local (grid*grid, 512)) for every example."""
    body = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    fc = model.fc.to(device).eval()
    globals_, locals_ = [], []
    batch = 64
    for start in range(0, len(examples), batch):
        images = torch.stack([transform(e.image) for e in examples[start : start + batch]]).to(
            device
        )
        feature_map = body(images)  # (B, 2048, 7, 7)
        pooled = F.adaptive_avg_pool2d(feature_map, 1) + F.adaptive_max_pool2d(feature_map, 1)
        globals_.append(F.normalize(fc(pooled.flatten(1)), dim=1).cpu())
        reduced = F.adaptive_avg_pool2d(feature_map, (grid, grid))  # (B, 2048, g, g)
        tokens = reduced.flatten(2).transpose(1, 2)  # (B, g*g, 2048)
        locals_.append(F.normalize(fc(tokens), dim=-1).cpu())
    return torch.cat(globals_), torch.cat(locals_)


def recall_at_1_global(vectors, labels):
    similarity = vectors @ vectors.T
    similarity.fill_diagonal_(-torch.inf)
    return float((labels[similarity.argmax(1)] == labels).float().mean())


def recall_at_1_maxsim(tokens, labels, device, chunk=32):
    n, t, d = tokens.shape
    flat = tokens.reshape(n * t, d).to(device)
    correct = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        query = tokens[start:stop].to(device).reshape(-1, d)  # (c*t, d)
        sim = query @ flat.T  # (c*t, n*t)
        sim = sim.reshape(stop - start, t, n, t)
        # MaxSim: best document token per query token, then average over query tokens.
        scored = sim.max(dim=3).values.mean(dim=1)  # (c, n)
        for row in range(stop - start):
            scored[row, start + row] = -torch.inf
        correct += int((labels[scored.argmax(1).cpu()] == labels[start:stop]).sum())
    return correct / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build()
    config = ImageEndToEndConfig(backbone_name="resnet50", head_pooling="avg_max", input_size=224)
    transform = _default_transform_factory(config, False)
    examples = load_image_retrieval_examples(dataset_name=DATASET, split="test")
    labels = torch.tensor([e.label for e in examples])
    print(f"{DATASET}: {len(examples)} test images, grid {GRID}x{GRID} = {GRID * GRID} tokens")

    g, loc = embed(model, examples, transform, device, GRID)
    print(f"  global pooled  d=512          R@1 = {recall_at_1_global(g, labels):.4f}")
    print(f"  MaxSim late interaction       R@1 = {recall_at_1_maxsim(loc, labels, device):.4f}")


if __name__ == "__main__":
    main()
