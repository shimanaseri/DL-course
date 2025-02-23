# Sixth Homework

The homeworks in this repository cover a variety of deep learning models and techniques, including:

### **Homework 1: Control VAE**
This homework focuses on implementing a **Control VAE**. The tasks include:

- **Model Introduction:** Understanding and implementing a Variational Autoencoder (VAE) and Control VAE.
- **Model Implementation:** Implementing both the encoder and decoder components of a VAE, using the dSprites dataset.
- **Model Evaluation:** Evaluating the VAE by plotting the reconstruction loss and KL divergence. You will also compute the Fréchet Inception Distance (FID) score to assess the quality of generated images.
- **Control VAE Implementation:** Adding a PI controller to the VAE architecture to control the balance between reconstruction and latent space regularization.

### **Homework 2: Generative Adversarial Networks (GANs)**
This homework involves learning and implementing different GAN architectures:

- **Model Training on MNIST:** Training a basic GAN on the MNIST dataset for image generation.
- **Wasserstein GAN:** Implementing and evaluating a Wasserstein GAN (WGAN) for improved training stability in GANs.
- **Self-Supervised GAN:** Implementing a Self-Supervised GAN (SSGAN) that incorporates unsupervised learning to generate images.
- **Model Evaluation:** Using the Fréchet Inception Distance (FID) score to evaluate the generated images. Additionally, training graphs such as loss curves for the generator and discriminator are analyzed.

### **Model Architectures**
- **Generator and Discriminator Models:** Implementing architectures for both the generator and discriminator, including residual blocks and modifications for stable training.
- **Hyperparameters:** Exploring the effect of different hyperparameters and adjusting learning rates, optimizers, and loss functions to improve model performance.

## How to Run the Code

1. Clone this repository or download the notebook and script files.
2. Install the required libraries. You can install the dependencies using:

   ```bash
   pip install -r requirements.txt
