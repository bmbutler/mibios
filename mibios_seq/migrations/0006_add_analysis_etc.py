from django.db import migrations, models
import django.db.models.deletion
import mibios.models


class Migration(migrations.Migration):
    """
    Overhaul the ASV-abundance-taxonomy complex

    Except for the reverse ASV row deletion the migration is auto-generated.
    ASVs get deleted on reverse to ease testing and development by avoiding
    constaints issues with new data trying to live in the old table.  The old
    tables were never populated with data, so nothing can get lost.
    """

    dependencies = [
        ('mibios', '0009_snapshot_migrations'),
        ('mibios_seq', '0005_sample'),
    ]

    def reverse_delete_asv(apps, schema_editor):
        apps.get_model('mibios_seq', 'ASV').objects.all().delete()

    operations = [
        migrations.CreateModel(
            name='Abundance',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(blank=True, default='', help_text='project specific ASV/OTU identifier', max_length=50, verbose_name='project internal id')),
                ('count', models.PositiveIntegerField(editable=False, help_text='absolute abundance')),
            ],
        ),
        migrations.CreateModel(
            name='AnalysisProject',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('description', models.TextField(blank=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Taxonomy',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('taxid', models.PositiveIntegerField(unique=True, verbose_name='NCBI taxonomy id')),
                ('name', models.CharField(max_length=300, unique=True, verbose_name='taxonomic name')),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.RemoveField(
            model_name='taxon',
            name='history',
        ),
        migrations.AlterModelOptions(
            name='asv',
            options={'ordering': ('number',)},
        ),
        migrations.AddField(
            model_name='asv',
            name='sequence',
            field=models.CharField(default='', editable=False, max_length=300, unique=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='asv',
            name='number',
            field=models.PositiveIntegerField(blank=True, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='asv',
            name='taxon',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='mibios_seq.Taxonomy'),
        ),
        migrations.DeleteModel(
            name='Community',
        ),
        migrations.DeleteModel(
            name='Taxon',
        ),
        migrations.AddField(
            model_name='analysisproject',
            name='asv',
            field=models.ManyToManyField(editable=False, through='mibios_seq.Abundance', to='mibios_seq.ASV'),
        ),
        migrations.AddField(
            model_name='analysisproject',
            name='history',
            field=models.ManyToManyField(to='mibios.ChangeRecord'),
        ),
        migrations.AddField(
            model_name='abundance',
            name='asv',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.ASV'),
        ),
        migrations.AddField(
            model_name='abundance',
            name='project',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.AnalysisProject'),
        ),
        migrations.AddField(
            model_name='abundance',
            name='sequencing',
            field=models.ForeignKey(editable=False, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.Sequencing'),
        ),
        migrations.AlterUniqueTogether(
            name='abundance',
            unique_together={('asv', 'sequencing', 'project'), ('name', 'sequencing', 'project')},
        ),
        migrations.RunPython(migrations.RunPython.noop, reverse_delete_asv)
    ]
