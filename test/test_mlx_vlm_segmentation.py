import automation.mlx_vlm_segmentation as mlx_vlm_segmentation


def test_parse_segment_response_accepts_middle_dot_punctuation_romaji():
    content = (
        '[{"text":"呼気性","romaji":"kokikisei"},'
        '{"text":"・","romaji":"·"},'
        '{"text":"吸気性","romaji":"kyuukisei"}]'
    )

    assert mlx_vlm_segmentation.parse_segment_response(content) == [
        {"text": "呼気性", "romaji": "kokikisei"},
        {"text": "・", "romaji": "/"},
        {"text": "吸気性", "romaji": "kyuukisei"},
    ]
