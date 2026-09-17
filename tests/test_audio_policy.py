import pytest

from robot_heard.audio import AudioInputError, AudioInputPolicy, audio_policy_from_config


def test_audio_policy_defaults_match_whisper_v0_contract():
    policy = AudioInputPolicy()
    assert policy.target_sample_rate == 16000
    assert policy.require_mono is True


def test_audio_policy_constructor_rejects_non_v0_values():
    with pytest.raises(AudioInputError, match="requires target_sample_rate=16000"):
        AudioInputPolicy(target_sample_rate=8000)
    with pytest.raises(AudioInputError, match="requires require_mono=true"):
        AudioInputPolicy(require_mono=False)


def test_audio_policy_from_config_accepts_frozen_v0_values():
    policy = audio_policy_from_config(
        {
            "audio": {
                "target_sample_rate": 16000,
                "require_mono": True,
            }
        }
    )
    assert policy == AudioInputPolicy(target_sample_rate=16000, require_mono=True)


def test_audio_policy_rejects_missing_audio_mapping():
    with pytest.raises(AudioInputError, match="config key 'audio' must be a mapping"):
        audio_policy_from_config({})


@pytest.mark.parametrize(
    "audio_config, expected_message",
    [
        ({}, "audio.target_sample_rate must be an integer"),
        ({"target_sample_rate": "16000", "require_mono": True}, "must be an integer"),
        ({"target_sample_rate": 8000, "require_mono": True}, "requires audio.target_sample_rate=16000"),
        ({"target_sample_rate": 16000, "require_mono": False}, "requires audio.require_mono=true"),
    ],
)
def test_audio_policy_rejects_unsupported_audio_values(audio_config, expected_message):
    with pytest.raises(AudioInputError, match=expected_message):
        audio_policy_from_config({"audio": audio_config})
