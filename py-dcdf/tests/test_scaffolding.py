import dcdf


def test_a_rust_function():
    assert dcdf.sum_as_string(19, 23) == "42"


def test_a_python_function():
    assert dcdf.helloworld() == "hello dad!"
