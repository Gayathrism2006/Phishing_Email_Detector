from app import app, db, User
from werkzeug.security import check_password_hash

with app.app_context():
  
    test_email = "test_user_for_debug@example.com"
    test_password = "TestPass123!"

   
    existing = User.query.filter_by(email=test_email).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()

    client = app.test_client()

   
    signup_resp = client.post('/signup', data={
        'email': test_email,
        'password': test_password,
        'confirmPassword': test_password,
        'termsCheck': 'on'
    }, follow_redirects=False)

    print('Signup status code:', signup_resp.status_code)
    print('Signup headers:', signup_resp.headers.get('Location'))

    
    user = User.query.filter_by(email=test_email).first()
    if user:
        print('User created in DB:', user.email)
        print('Password stored hashed:', user.password != test_password)
        print('check_password_hash OK:', check_password_hash(user.password, test_password))
    else:
        print('User not found in DB after signup')

 
    login_resp = client.post('/login', data={
        'email': test_email,
        'password': test_password
    }, follow_redirects=False)

    print('Login status code:', login_resp.status_code)
    print('Login headers:', login_resp.headers.get('Location'))

    
    if login_resp.status_code == 200:
        body = login_resp.get_data(as_text=True)
        print('\nLogin response body snippet:\n', body[:400])

    
    if user:
        db.session.delete(user)
        db.session.commit()
        print('Test user removed')
