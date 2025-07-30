import pytest

def test_successful_deserialization():
    allow_dangerous_deserialization = True
    # Simulate loading a pickle file
    result = load_pickle_file("trusted_file.pkl", allow_dangerous_deserialization)
    assert result is not None

def test_error_handling_deserialization():
    allow_dangerous_deserialization = False
    with pytest.raises(ValueError):
        load_pickle_file("untrusted_file.pkl", allow_dangerous_deserialization)