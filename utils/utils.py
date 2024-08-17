import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_training_params(args):
    if args.scale:
        return {
            'img_scale': args.scale
        }
    elif args.size:
        return {
            'imgW': args.size[0],
            'imgH': args.size[1]
        }
    else:
        # Default image dimensions
        return {
            'imgW': 224,
            'imgH': 224
        }
