import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from feature_extractor import extract_features
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
app.secret_key = "parkinson_secret_2025"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

rf_model = None
x_train, x_test, y_train, y_test = None, None, None, None

FEATURE_NAMES = [
    'std_6th_delta_delta','std_8th_delta_delta','std_9th_delta_delta',
    'app_entropy_shannon_2_coef','app_entropy_shannon_3_coef','app_entropy_shannon_4_coef',
    'app_entropy_shannon_5_coef','app_entropy_shannon_6_coef','app_entropy_shannon_7_coef',
    'app_entropy_shannon_8_coef','app_entropy_shannon_9_coef','app_entropy_shannon_10_coef',
    'app_entropy_log_3_coef','app_entropy_log_5_coef','app_entropy_log_6_coef',
    'app_entropy_log_7_coef','app_entropy_log_8_coef','app_entropy_log_9_coef',
    'app_entropy_log_10_coef','app_TKEO_std_9_coef','app_TKEO_std_10_coef',
    'app_LT_entropy_shannon_4_coef','app_LT_entropy_shannon_5_coef',
    'app_LT_entropy_shannon_6_coef','app_LT_entropy_shannon_7_coef',
    'app_LT_entropy_shannon_8_coef','app_LT_entropy_shannon_9_coef',
    'app_LT_entropy_shannon_10_coef','app_LT_entropy_log_3_coef',
    'app_LT_entropy_log_4_coef','app_LT_entropy_log_5_coef','app_LT_entropy_log_6_coef',
    'app_LT_entropy_log_7_coef','app_LT_entropy_log_8_coef','app_LT_entropy_log_9_coef',
    'app_LT_entropy_log_10_coef','app_LT_TKEO_mean_8_coef','app_LT_TKEO_std_7_coef',
    'tqwt_entropy_log_dec_35','tqwt_TKEO_mean_dec_7'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_train_model():
    global rf_model, x_train, x_test, y_train, y_test

    df = pd.read_csv('pd_speech_features.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_selected = X_res[FEATURE_NAMES]

    x_train, x_test, y_train, y_test = train_test_split(
        X_selected, y_res, test_size=0.3, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(x_train, y_train)

load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':

        mode = request.form.get('input_mode', 'audio')

        try:
            if mode == 'audio':
                if 'audio' not in request.files:
                    flash("No audio file uploaded.", "danger")
                    return render_template('prediction.html')

                audio_file = request.files['audio']
                if audio_file.filename == '':
                    flash("Please select a voice file.", "danger")
                    return render_template('prediction.html')

                if not allowed_file(audio_file.filename):
                    flash("Unsupported file type. Use .wav, .mp3, .ogg or .m4a", "danger")
                    return render_template('prediction.html')

                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)

                values = extract_features(filepath)

                try:
                    os.remove(filepath)
                except Exception:
                    pass

            else:
                values = [float(request.form[f]) for f in FEATURE_NAMES]

            # Predict
            input_data = np.array([values])
            pred       = rf_model.predict(input_data)[0]
            prob       = rf_model.predict_proba(input_data)[0]
            confidence = round(max(prob) * 100, 2)  # ✅ No fake +10

            if pred == 1:
                if confidence >= 85:
                    result = "Affected by Parkinson's Disease"
                    color  = "danger"
                elif confidence >= 70:
                    result = "Possibly Affected — Consult a Doctor"
                    color  = "warning"
                else:
                    result = "Inconclusive — Please Retake the Test"
                    color  = "secondary"
            else:
                if confidence >= 85:
                    result = "Healthy"
                    color  = "success"
                elif confidence >= 70:
                    result = "Likely Healthy — Monitor if Symptoms Persist"
                    color  = "info"
                else:
                    result = "Inconclusive — Please Retake the Test"
                    color  = "secondary"

            session.pop('result', None)
            session.pop('messages', None)
            session['result'] = {'text': result, 'confidence': confidence}

            return render_template('prediction.html',
                                   prediction=result,
                                   confidence=confidence,
                                   color=color)

        except KeyError as e:
            flash(f"Missing field: {e}", "danger")
        except ValueError as e:
            flash(f"Invalid value: {e}", "danger")
        except Exception as e:
            flash(f"Error processing input: {e}", "danger")

    return render_template('prediction.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    result_data = session.get('result')

    if not result_data:
        flash("Please complete a prediction first!", "warning")
        return redirect(url_for('prediction'))

    if 'messages' not in session:
        session['messages'] = []

    if len(session['messages']) == 0:
        welcome = f"""
Hello! I am your AI Assistant Doctor.

Result: {result_data['text']}
Confidence: {result_data['confidence']}%

This is NOT a final diagnosis.
"""
        session['messages'].append({"role": "assistant", "content": welcome.strip()})
        session.modified = True

    if request.method == 'POST':
        user_msg = request.form.get('question', '').strip()

        if user_msg:
            session['messages'].append({"role": "user", "content": user_msg})
            session.modified = True

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""You are a medical AI assistant helping a 
patient understand their Parkinson's Disease screening result.
Patient Result: {result_data['text']}
Confidence: {result_data['confidence']}%
Always remind the user this is NOT a final diagnosis and they should consult a doctor."""
                        },
                        {
                            "role": "user",
                            "content": user_msg
                        }
                    ]
                )
                reply = response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str:
                    reply = "⚠️ AI is temporarily busy. Please wait 30 seconds and try again."
                else:
                    reply = "⚠️ Something went wrong. Please try again shortly."

            session['messages'].append({"role": "assistant", "content": reply})
            session.modified = True

    return render_template('chatbot.html', messages=session['messages'])


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    msg = None
    if request.method == 'POST':
        choice = request.form.get('Algorithm')

        if choice == '1':
            rf = RandomForestClassifier(n_estimators=200)
            rf.fit(x_train, y_train)
            acc = accuracy_score(y_test, rf.predict(x_test)) * 100
            msg = f"Random Forest → {acc:.2f}%"

        elif choice == '2':
            svm = SVC()
            svm.fit(x_train, y_train)
            acc = accuracy_score(y_test, svm.predict(x_test)) * 100
            msg = f"SVM → {acc:.2f}%"

    return render_template('model.html', accuracy=msg)


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/new_test')
def new_test():
    session.pop('result', None)
    session.pop('messages', None)
    return redirect(url_for('prediction'))


@app.route('/test_predict/<sample_type>')
def test_predict(sample_type):
    if sample_type == 'parkinson':
        values = [0.016163, 0.022442, 0.020834,
                  -80587116.9, -106534630.9, -154226817.9,
                  -250154463.1, -447073332.2, -882097599.2,
                  -1729776667.0, -3611295007.0, -7526069553.0,
                  468.687, 274.4553, 245.0825, 241.7578,
                  237.0451, 247.4423, 257.8395,
                  5140080.116, 10288111.65,
                  -66095.327, -113376.156, -212607.801,
                  -437553.8472, -890429.1532, -1920892.641,
                  -4121853.861, 203.2454, 157.0924,
                  134.7072, 126.287, 129.949, 132.2244,
                  142.6216, 153.0188, 898.3043, 1149.7919,
                  -3147.4035, 0.00000925]
    else:
        values = [0.014831, 0.012968, 0.0099697,
                  -279956270.2, -367615417.4, -528975986.2,
                  -853605045.9, -1518599373.0, -2985222650.0,
                  -5836728770.0, -12150848615.0, -25256435889.0,
                  512.3713, 297.3978, 264.5699, 260.0941,
                  254.2373, 264.6346, 275.0318,
                  16122022.65, 32197448.47,
                  -84592.7727, -144566.5653, -270277.562,
                  -554890.4961, -1126984.065, -2427045.001,
                  -5200243.433, 211.3209, 162.6104,
                  138.9479, 129.8889, 133.3382, 135.4021,
                  145.7993, 156.1965, 1108.8345, 1420.6598,
                  -2589.3281, 0.0002032]

    input_data = np.array([values])
    pred = rf_model.predict(input_data)[0]
    prob = rf_model.predict_proba(input_data)[0]
    confidence = round(max(prob) * 100, 2)
    result = "Affected by Parkinson's Disease" if pred == 1 else "Healthy"
    color = "danger" if pred == 1 else "success"

    session.pop('result', None)
    session.pop('messages', None)
    session['result'] = {'text': result, 'confidence': confidence}

    return render_template('prediction.html',
                           prediction=result,
                           confidence=confidence,
                           color=color)


@app.route('/view')
def view():
    df = pd.read_csv('test.csv')
    dummy = df.head(100).to_html()
    return render_template('view.html', data=dummy)
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))