import random
import string
import pandas as pd
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# Generate random plaintext
def generate_plaintext(length=32):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)).encode()

# Encrypt with AES
def encrypt_aes(plaintext, key_size=16):
    key = get_random_bytes(key_size)  # Key size: 16 (128-bit), 24 (192-bit), or 32 (256-bit)
    cipher = AES.new(key, AES.MODE_ECB)
    padded_plaintext = plaintext + b' ' * (16 - len(plaintext) % 16)  # Pad to multiple of 16 bytes
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext.hex()

# Encrypt with DES
def encrypt_des(plaintext):
    key = get_random_bytes(8)  # 64-bit key
    cipher = DES.new(key, DES.MODE_ECB)
    padded_plaintext = plaintext + b' ' * (8 - len(plaintext) % 8)  # Pad to multiple of 8 bytes
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext.hex()

# Encrypt with Blowfish
def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)  # Key size can range from 4 to 56 bytes
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    padded_plaintext = plaintext + b' ' * (8 - len(plaintext) % 8)  # Pad to multiple of 8 bytes
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext.hex()

# Encrypt with ChaCha20
def encrypt_chacha20(plaintext):
    key = get_random_bytes(32)  # 256-bit key
    nonce = get_random_bytes(16)  # ChaCha20 requires a 128-bit (16-byte) nonce
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None)
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext)
    return ciphertext.hex()

# Generate dataset
def generate_dataset(num_samples=3000):
    data = {"Ciphertext": [], "Algorithm": []}

    for _ in range(num_samples):
        plaintext = generate_plaintext()

        # AES
        aes_cipher = encrypt_aes(plaintext)
        data["Ciphertext"].append(aes_cipher)
        data["Algorithm"].append("AES")

        # DES
        des_cipher = encrypt_des(plaintext)
        data["Ciphertext"].append(des_cipher)
        data["Algorithm"].append("DES")

        # Blowfish
        blowfish_cipher = encrypt_blowfish(plaintext)
        data["Ciphertext"].append(blowfish_cipher)
        data["Algorithm"].append("Blowfish")

        # ChaCha20
        chacha20_cipher = encrypt_chacha20(plaintext)
        data["Ciphertext"].append(chacha20_cipher)
        data["Algorithm"].append("ChaCha20")

    return data

# Save to CSV
if __name__ == "__main__":
    num_samples = 5000
    dataset = generate_dataset(num_samples)
    df = pd.DataFrame(dataset)
    df.to_csv("encryption_dataset_aes_des_blowfish_chacha20.csv", index=False)
    print(f"Dataset with {num_samples * 4} entries saved as 'encryption_dataset_aes_des_blowfish_chacha20.csv'")
