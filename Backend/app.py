from flask import Flask, render_template, request, redirect, url_for, flash
from flask import jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
from methods import calculateEmbedding, searchPeople, findCosineDistance


app = Flask(__name__)
# SQLite database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './uploads'
db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class User(db.Model):
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    enroll_number = db.Column(db.String(20), unique=True, primary_key=True)
    password = db.Column(db.String(100))
    batch = db.Column(db.String(50))
    course = db.Column(db.String(100))
    image_filename = db.Column(db.String(100))

    def __repr__(self):
        return f'<User {self.enroll_number}>'

# create another model to save facial embeddings along with enrollment number


class FacialEmbedding(db.Model):
    enroll_number = db.Column(db.String(20), unique=True, primary_key=True)
    # embedding could be very large arrays, so use appropriate data type like pickle
    embeddings = db.Column(db.PickleType)

    def __repr__(self):
        return f'<FacialEmbedding {self.enroll_number}>'


@app.route('/')
def index():
    return jsonify({'message': 'Hello World'})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/student-signup', methods=['POST'])
def signup():
    data = request.form  # Access form data
    enroll_number = data.get('enroll_number')

    existing_user = User.query.filter_by(enroll_number=enroll_number).first()
    if existing_user:
        return jsonify({'message': 'User already exists'}), 400

    hashed_password = generate_password_hash(
        data.get('password'), method='pbkdf2:sha256', salt_length=8)

    # Check if an image file is included in the request

    if '' in request.files:
        image = request.files['']
        print(image)
        if image.filename != '' and allowed_file(image.filename):
            # Save the uploaded image with the user's enroll number as the filename
            image_filename = enroll_number + \
                os.path.splitext(image.filename)[1]
            image.save(os.path.join(
                app.config['UPLOAD_FOLDER'], image_filename))
            print(image_filename, "saved")

            # calculate embedding and save it to database along with enroll number
            embedding = calculateEmbedding(
                os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
            new_embedding = FacialEmbedding(
                enroll_number=enroll_number, embeddings=embedding)
            db.session.add(new_embedding)
            print("Embedding saved!")
            db.session.commit()

        else:
            image_filename = None
    else:
        image_filename = None
        print("No file found")

    new_user = User(
        full_name=data.get('full_name'),
        email=data.get('email'),
        enroll_number=enroll_number,
        password=hashed_password,
        batch=data.get('batch'),
        course=data.get('course'),
        image_filename=image_filename

    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 200


@app.route('/login', methods=['POST'])
def login():
    data = request.form
    enroll_number = data.get('enroll_number')
    password = data.get('password')

    user = User.query.filter_by(enroll_number=enroll_number).first()
    if not user:
        return jsonify({'message': 'User does not exist'})

    if not check_password_hash(user.password, password):
        return jsonify({'message': 'Incorrect password'})

    print("Login successful")
    return jsonify({'message': 'Login successful'})


def create_tables():
    with app.app_context():
        db.create_all()


def findpeople(groupEmb):
    threshold = 0.4
    presentees = []
    for row in FacialEmbedding.query.all():
        enroll_number = row.enroll_number
        embedding = row.embeddings
        for faceEmb in groupEmb:
            res = findCosineDistance(embedding, faceEmb)
            if res <= threshold:
                presentees.append(enroll_number)
                break
    return presentees


# create an api that will take picture and return the enroll number of the students
@app.route('/search', methods=['POST'])
def search():
    data = request.form  # Access form data
    date = data.get('date')
    batch = data.get('batch')
    course = data.get('course')
    print(date, batch, course)
    if '' in request.files:
        image = request.files['']
        print(image)
        # save this image temporarily to a folder and then delete it after processing.
        # This will save storage space

        if image.filename != '' and allowed_file(image.filename):
            # this is a group image so save it temporarily
            image_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], image.filename.split('.')[0] + '_temp' + os.path.splitext(image.filename)[1])
            image.save(image_filename)
            print(image_filename, "saved")
            # give url of image to searchPeople function
            grpEmbs = searchPeople(image_filename)
        # delete the image after processing
        os.remove(image_filename)
        print("Image deleted")
        # finding people
        people = findpeople(grpEmbs)
    return jsonify({'message': people})


if __name__ == '__main__':
    create_tables()
    # run the app on http://localhost:5005
    app.run(debug=True, port=5005)
