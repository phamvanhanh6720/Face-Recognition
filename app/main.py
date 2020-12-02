from app import app
from flask import render_template, redirect
from flask import request, url_for
from face_verification import FaceVerification
from werkzeug.utils import secure_filename
import os
import cv2
from PIL.Image import Image
import time

face_verifier = FaceVerification()
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]


@app.route("/")
def index():
    return redirect('/upload_image')


def allowed_image(filename):

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route("/upload_image", methods=["GET", "POST"])
def compare():
    if request.method == "POST":
        if request.files:
            image1 = request.files["image1"]
            image2 = request.files["image2"]

            if image1.filename == "" or image2.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image1.filename) and allowed_image(image2.filename):
                filename1 = secure_filename(image1.filename)
                filename2 = secure_filename(image2.filename)

                filepath1 = os.path.join(app.config["IMAGE_UPLOADS"], filename1)
                filepath2 = os.path.join(app.config["IMAGE_UPLOADS"], filename2)
                image1.save(filepath1)
                image2.save(filepath2)
                print("Image saved")

                image_1 = cv2.imread(filepath1)
                image_2 = cv2.imread(filepath2)
                cosin, distance, facial_image1, facial_image2 = face_verifier.calculate_similarity(image_1, image_2)

                cv2.imwrite('./app/static/faces/' + filename1, cv2.cvtColor(facial_image1, cv2.COLOR_RGB2BGR))
                cv2.imwrite('./app/static/faces/' + filename2, cv2.cvtColor(facial_image2, cv2.COLOR_RGB2BGR))
                similarity = "{:.2f} %".format(cosin * 100)
                print("distance: ", distance)
                print("cosin: ", cosin)

                return render_template(
                    "public/result.html", filepath1=os.path.join('./static/faces', filename1) + "?" + str(time.time()),
                    filepath2=os.path.join('./static/faces', filename2) + "?" + str(time.time()), similarity=similarity)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("public/upload_image.html")


