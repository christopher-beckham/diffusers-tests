import pytest

# Setup function that runs only once before all tests in the module and returns an object
@pytest.fixture(scope="module")
def setup_function():
    print("Setting up before all tests.")
    # Example of a resource-intensive object (could be a database connection, large dataset, etc.)
    obj = {"key": "value", "number": 42}
    
    # Yield the object so it can be passed to the tests
    yield obj
    
    print("Tearing down after all tests.")
    # Add resource cleanup logic here if necessary

# Test function that uses the object from the setup function
def test_assert_true(setup_function):
    obj = setup_function
    assert obj["number"] == 42, "Test should pass because the number is 42"

# Test function that uses the object and asserts a false condition
def test_assert_false(setup_function):
    obj = setup_function
    assert obj["number"] != 100, "Test should fail because the number is not 100"

# Another test using the object
def test_another_true(setup_function):
    obj = setup_function
    with pytest.raises(Exception):
        raise Exception()

# Another test using the object and asserting false
def test_another_false(setup_function):
    obj = setup_function
    assert obj["key"] != "other_value", "Test should fail because the key is not 'other_value'"

if __name__ == "__main__":
    pytest.main()