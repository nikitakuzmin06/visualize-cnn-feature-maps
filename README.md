# visualize-cnn-feature-maps

A small notebook that hooks into a pretrained VGG19 and looks at what its
intermediate layers actually produce when given an image.

## Why

CNNs are usually treated as black boxes — image in, prediction out. This
project pokes at the middle of that box: it grabs the raw output tensors of a
few `VGG19` layers using PyTorch forward hooks, and renders them as images so
you can see what early vs. late layers are responding to.

It's a learning exercise, not a tool or library.

## What's been done

- Loaded a pretrained `torchvision.models.vgg19`.
- Attached forward hooks to four layers in `model.features`:
  - `layer 3` and `layer 4` (early layers, low-level features)
  - `layer 21` and `layer 22` (deep layers, high-level features)
- Ran a forward pass and confirmed the captured tensor shapes:

  | Layer | Shape (N, C, H, W) |
  |-------|--------------------|
  | 3     | `[1, 64, 224, 224]` |
  | 4     | `[1, 64, 112, 112]` |
  | 21    | `[1, 512, 28, 28]`  |
  | 22    | `[1, 512, 28, 28]`  |

- Loaded a real image (`augustus.jpeg`), preprocessed it to ImageNet/VGG19
  standards (resize to 224x224, normalize with ImageNet mean/std).
- Visualized a single feature map channel from `layer 22` with matplotlib.
- Found the channel with the highest total activation (summed over H and W)
  in `layer 22` and plotted it — turned out to be channel `159`.

## Example

Input: `augustus.jpeg`

Output: a grayscale plot of a single 28x28 feature map from `layer 22`
(the most active channel for this image), showing the spatial pattern that
channel reacts to.

## Scope

- Single notebook (`main.ipynb`), no scripts, no CLI, no package.
- Only VGG19 is used, and only 4 of its 36 `features` layers are hooked.
- Visualization is per-channel grayscale only — no channel grids, no
  Grad-CAM, no saliency maps.
- Input is a single hardcoded image.

This is meant as a starting point for exploring CNN internals, not a
finished visualization tool.

## Requirements

- Python 3
- `torch`
- `torchvision`
- `Pillow`
- `matplotlib`

## Usage

Open `main.ipynb` in Jupyter and run the cells top to bottom. The first run
will download VGG19's pretrained weights.

## License

MIT — see [LICENSE](LICENSE).
