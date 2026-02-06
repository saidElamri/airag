import auth

def test_password_hashing():
    password = "secret_password"
    hashed = auth.get_password_hash(password)
    assert hashed != password
    assert auth.verify_password(password, hashed) is True
    assert auth.verify_password("wrong_password", hashed) is False

def test_jwt_token_creation():
    data = {"sub": "test@example.com"}
    token = auth.create_access_token(data)
    assert token is not None
    assert isinstance(token, str)
    
    # Test decoding
    from jose import jwt
    payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
    assert payload["sub"] == "test@example.com"
