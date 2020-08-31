from django.db import migrations, models
import mibios.models


class Migration(migrations.Migration):

    dependencies = [
        ('mibios', '0006_snapshot_jsondump'),
        ('hhcd', '0008_sequencing_transition'),
        ('mibios_seq', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='SeqNote',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('text', models.TextField(blank=True, max_length=5000)),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.RenameField('sequencing', 'note', 'oldnote'),
        migrations.AddField(
            model_name='sequencing',
            name='note',
            field=models.ManyToManyField(blank=True, to='mibios_seq.SeqNote'),
        ),
    ]
