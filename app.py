from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import secrets
import json
import re
from sklearn.feature_extraction import text
from langdetect import detect
from translate import Translator


app = Flask(__name__)

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or 'dev_secret_change_me'


db_path = os.path.join(app.root_path, 'users.db')
db_uri = 'sqlite:///' + db_path.replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri

print(f"[app] Using SQLite DB at: {db_path}")

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(300), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    detected_language = db.Column(db.String(10), nullable=True)
    translated_text = db.Column(db.Text, nullable=True)
    trusted_source = db.Column(db.Boolean, default=False)
    user = db.relationship('User', backref='detections')


model = joblib.load("binary_email_classifier.pkl")
vectorizer = joblib.load("binary_email_tfidf_vectorizer.pkl")


stop_words = text.ENGLISH_STOP_WORDS



def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  

def translate_to_english(text, source_lang):
    if source_lang == 'en':
        return text 
    try:

        translator = Translator(from_lang=source_lang, to_lang='en')
       
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text 


def preprocess_email(text_in):
   
    text_in = text_in.lower()
    
   
    legitimate_indicators = ['booking', 'confirmation', 'ticket', 'flight', 'reservation', 'itinerary', 'passenger']
    
   
    pattern = '|'.join([r'\b' + word + r'\b' for word in legitimate_indicators])
    
    
    matches = re.findall(pattern, text_in)
    
   
    text_in = re.sub(r'[^a-zA-Z\s]', ' ', text_in)
    
    
    text_in = re.sub(r'\s+', ' ', text_in).strip()
    
 
    text_in = ' '.join([word for word in text_in.split() if word not in stop_words])
    
    
    if matches:
        legitimate_text = ' '.join(matches) + ' ' + ' '.join(matches)
        text_in = text_in + ' ' + legitimate_text
    
    return text_in


BEST_THRESHOLD = 0.58


@app.route('/')
def home():
    if 'user' in session:
        user = User.query.filter_by(email=session['user']).first()
        unsafe_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Unsafe').count()
        ham_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Ham').count()

        return render_template(
            'index.html',
            prediction=None,
            subject=None,
            fake_count=json.dumps(unsafe_count),
            real_count=json.dumps(ham_count)
        )
    return render_template('home.html', show_login=False, error=None)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')
        terms_check = request.form.get('termsCheck')
        
        
        if not email or not password or not confirm_password:
            error = "All fields are required!"
            return render_template('signup.html', error=error)
            
        if password != confirm_password:
            error = "Passwords do not match!"
            return render_template('signup.html', error=error)
            
        if len(password) < 8:
            error = "Password must be at least 8 characters long!"
            return render_template('signup.html', error=error)
            
        if not terms_check:
            error = "You must agree to the terms and conditions!"
            return render_template('signup.html', error=error)
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            error = "Email already exists!"
            return render_template('signup.html', error=error)
        else:
            hashed_pw = generate_password_hash(password)
            new_user = User(email=email, password=hashed_pw)
            db.session.add(new_user)
            db.session.commit()
  
            print(f"[signup] Created user: {email}")
            return redirect(url_for('login', email=email))
    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    
    if request.method == 'GET':
        prefill_email = request.args.get('email')
    else:
        prefill_email = request.form.get('email')

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember_me = request.form.get('rememberMe')

        
        if not email or not password:
            error = "Email and password are required!"
            return render_template('login.html', error=error, prefill_email=prefill_email)

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user'] = email

          
            if remember_me:
                session.permanent = True  
            print(f"[login] Successful login for: {email}")
            return redirect(url_for('home'))
        else:
            print(f"[login] Failed login attempt for: {email}")
            error = "Invalid credentials!"

    return render_template('login.html', error=error, prefill_email=prefill_email)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


TRUSTED_DOMAINS = [
    'indigo.in',     
    'goindigo.in',   
    'airindia.in',    
    'spicejet.com',   
    'airasia.com',   
    'vistara.com',    
    'booking.com',   
    'makemytrip.com', 
    'cleartrip.com', 
    'yatra.com',      
    'expedia.com',    
    'hotels.com',     
    'airbnb.com',     
    'irctc.co.in'     
]


def extract_sender_domain(email_content):

    email_pattern = r'[Ff]rom:\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    match = re.search(email_pattern, email_content)
    
    if match:
        email_address = match.group(1)
       
        domain = email_address.split('@')[-1].lower()
        return domain
    return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    subject = request.form['subject']
    translate_option = request.form.get('translate_option', 'auto')

    # Validate input
    if not subject or len(subject.strip()) == 0:
        # Counts for dashboard
        user = User.query.filter_by(email=session['user']).first()
        unsafe_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Unsafe').count()
        ham_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Ham').count()
        
        return render_template(
            'index.html',
            error="Please enter email content to analyze",
            prediction=None,
            subject=None,
            fake_count=json.dumps(unsafe_count),
            real_count=json.dumps(ham_count)
        )
    
   
    sender_domain = extract_sender_domain(subject)
    is_trusted = False
    if sender_domain:
        is_trusted = any(trusted_domain in sender_domain for trusted_domain in TRUSTED_DOMAINS)
    
   
    detected_language = detect_language(subject)
    original_text = subject
    translated_text = None
    
    
    if detected_language != 'en' and translate_option != 'no_translate':
        translated_text = translate_to_english(subject, detected_language)
     
        text_for_prediction = translated_text
    else:
       
        text_for_prediction = subject


    if is_trusted:
        prediction = "Ham"
        proba = 0.1 
        confidence_score = 1 - proba 
    else:
         
        clean_subject = preprocess_email(text_for_prediction)
        subject_vec = vectorizer.transform([clean_subject])

       
        proba = model.predict_proba(subject_vec)[0][1]
        prediction_numeric = int(proba >= BEST_THRESHOLD)
        prediction = "Unsafe" if prediction_numeric == 1 else "Ham"
      
        confidence_score = proba if prediction == "Unsafe" else 1 - proba
    
  
    confidence_percentage = round(confidence_score * 100, 2)

   
    user = User.query.filter_by(email=session['user']).first()
    record = DetectionHistory(
        user_id=user.id, 
        subject=subject, 
        prediction=prediction,
        detected_language=detected_language,
        translated_text=translated_text,
        trusted_source=is_trusted
    )
    db.session.add(record)
    db.session.commit()

    unsafe_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Unsafe').count()
    ham_count = DetectionHistory.query.filter_by(user_id=user.id, prediction='Ham').count()

    return render_template(
        'index.html',
        prediction=prediction,
        subject=subject,
        confidence=confidence_percentage,
        detected_language=detected_language,
        translated_text=translated_text,
        trusted_source=is_trusted,
        sender_domain=sender_domain,
        fake_count=json.dumps(unsafe_count),
        real_count=json.dumps(ham_count)
    )

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(email=session['user']).first()
    detections = DetectionHistory.query.filter_by(user_id=user.id).order_by(DetectionHistory.timestamp.desc()).all()
    return render_template('history.html', detections=detections, detection_id=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)






