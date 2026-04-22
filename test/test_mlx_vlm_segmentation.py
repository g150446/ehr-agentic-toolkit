import automation.mlx_vlm_segmentation as mlx_vlm_segmentation


def test_parse_segment_response_accepts_middle_dot_punctuation_romaji():
    content = (
        '[{"text":"呼気性"},'
        '{"text":"・"},'
        '{"text":"吸気性"}]'
    )

    assert mlx_vlm_segmentation.parse_segment_response(content) == [
        {"text": "呼気性", "romaji": "kokisei"},
        {"text": "・", "romaji": "/"},
        {"text": "吸気性", "romaji": "kyuukisei"},
    ]


def test_parse_segment_response_ignores_vlm_romaji_and_uses_local_override():
    content = '[{"text":"日前","romaji":"zenjitsu"}]'

    assert mlx_vlm_segmentation.parse_segment_response(content) == [
        {"text": "日前", "romaji": "nichimae"},
    ]
