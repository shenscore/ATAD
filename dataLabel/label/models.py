from django.db import models
import slugid


def decoded_slugid():
    return slugid.nice()


# mcool data
class Mcool(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank= True)
    datafile = models.FileField(upload_to='mcool')
    uid = models.CharField(max_length=20, unique = True, default = decoded_slugid)


# feature dict data
class FeatureDict(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank= True)
    datafile = models.FileField(upload_to='featureDicts')
    uid = models.CharField(max_length=20, unique = True, default = decoded_slugid)
    mcool = models.ForeignKey(Mcool, models.CASCADE)

