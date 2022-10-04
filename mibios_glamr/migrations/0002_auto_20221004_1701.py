# Generated by Django 2.2.26 on 2022-10-04 21:01

from django.db import migrations
import mibios_umrad.fields


class Migration(migrations.Migration):

    dependencies = [
        ('mibios_glamr', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='reference',
            name='reference_id',
            field=mibios_umrad.fields.AccessionField(default='foo', prefix='paper_'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='dataset',
            name='dataset_id',
            field=mibios_umrad.fields.AccessionField(help_text='GLAMR accession to data set/study/project', verbose_name='Dataset ID'),
        ),
        migrations.AlterUniqueTogether(
            name='reference',
            unique_together=set(),
        ),
    ]
