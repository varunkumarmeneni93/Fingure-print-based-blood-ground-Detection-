from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("TrainML", views.TrainML, name="TrainML"),
	       path("Predict.html", views.Predict, name="Predict"),
               path("PredictAction", views.PredictAction, name="PredictAction"),	       
]
