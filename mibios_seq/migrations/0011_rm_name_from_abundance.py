# Generated by Django 2.2.14 on 2020-10-08 19:32
# and modified and renames

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mibios_seq', '0010_re_design_otu_model'),
    ]

    def reverse(apps, schema_editor):
        """
        Re-populate names from OTU names

        Should be run before uniqueness contraint is re-raised
        """
        Abundance = apps.get_model('mibios_seq', 'Abundance')
        qs = Abundance.objects.select_related('otu')
        for i in qs:
            i.name = i.otu.prefix + str(i.otu.number)
        qs.bulk_update(qs, fields=['name'], batch_size=10000)

    operations = [
        migrations.AlterField(
            model_name='abundance',
            name='otu',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.OTU', verbose_name='OTU'),
        ),
        migrations.AlterField(
            model_name='abundance',
            name='project',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.AnalysisProject', verbose_name='analysis project'),
        ),
        migrations.AlterUniqueTogether(
            name='abundance',
            unique_together={('otu', 'sequencing', 'project')},
        ),
        migrations.RunPython(migrations.RunPython.noop, reverse),
        migrations.RemoveField(
            model_name='abundance',
            name='name',
        ),
    ]
