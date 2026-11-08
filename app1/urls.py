from django.urls import path
from . import views
urlpatterns = [
 path('',views.home,name='home'),path('login/',views.login_view,name='login'),
 path('signup/',views.signup,name='signup'),path('logout/',views.logout_view,name='logout'),
 path('images/<int:pk>/',views.detail,name='detail'),
 path('images/<int:pk>/original/',views.original,name='original'),
 path('images/<int:pk>/detect/',views.detect,name='detect'),
 path('images/<int:pk>/feedback/',views.feedback,name='feedback'),
 path('results/<uuid:token>/<str:name>/',views.artifact,name='artifact'),
]
