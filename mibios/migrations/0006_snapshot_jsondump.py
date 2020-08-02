# Generated by Django 2.2.13 on 2020-08-01 15:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mibios', '0005_auto_20200731_1122'),
    ]

    operations = [
        migrations.AddField(
            model_name='snapshot',
            name='jsondump',
            field=models.FilePathField(default='???', editable=False, path='/geomicro/data2/heinro/src/mibios/snapshots', verbose_name='JSON formatted archive'),
            preserve_default=False,
        ),
    ]
