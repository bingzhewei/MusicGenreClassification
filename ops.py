import numpy as np


def combine_audio_noise(audio_tensor,
                        noise_tensor,
                        noise_level=None,
                        normalize=False):
    """
    Take an audio tensor, a noise tensor, noise level (the ratio of noise/audio).
    Return processed original audio (ground truth) and combined audio (added noise)
    Args:
        audio_tensor: the good audio
        noise_tensor: the bad noise
        noise_level: the ratio of noise / audio in the synthesis. By default the noise level will be a random value between 0.1 and 0.3
        normalize: if output should be normalized (to avoid clipping, etc.)
    """
    noise_level = np.random.rand(
    ) * 0.2 + 0.1 if noise_level is None else noise_level

    if normalize:
        audio_output = (1 / (1 + noise_level)) * audio_tensor
        combined_output = audio_output + (noise_level /
                                          (1 + noise_level)) * noise_tensor
    else:
        audio_output = audio_tensor
        combined_output = audio_output + noise_level * noise_tensor
    return audio_output, combined_output
