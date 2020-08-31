from django.db import migrations, models

import mibios


class Migration(migrations.Migration):
    """
    Create and populate the prospective parent Sample model.

    Switch Sequencing.sample to the parent. After this, FecalSample can be made
    the child of Sample in hhcd.
    """

    dependencies = [
        ('mibios', '0006_snapshot_jsondump'),
        ('mibios_seq', '0004_remove_sequencing_oldnote'),
    ]

    def forward(apps, schema_editor):
        """
        1. Create one Sample object per FecalSample with same pk
        2. move history from FecalSample to Sample
        """
        FecalSample = apps.get_model('hhcd', 'FecalSample')
        Sample = apps.get_model('mibios_seq', 'Sample')
        Sample.objects.bulk_create(
            [Sample(id=i) for i in FecalSample.objects.values_list('pk', flat=True)]
        )
        for i in Sample.objects.all():
            fs = FecalSample.objects.prefetch_related('history').get(pk=i.pk)
            i.history.set(fs.history.all())
            fs.history.clear()

    def reverse(apps, schema_editor):
        """
        move history from Sample back to FecalSample
        """
        FecalSample = apps.get_model('hhcd', 'FecalSample')
        Sample = apps.get_model('mibios_seq', 'Sample')
        for i in FecalSample.objects.all():
            s = Sample.objects.prefetch_related('history').get(pk=i.pk)
            i.history.set(s.history.all())
            s.history.clear()

    operations = [
        migrations.CreateModel(
            name='Sample',
            fields=[
                ('id', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.RunPython(forward, reverse),
        migrations.AlterField(
            model_name='Sample',
            name='id',
            field=mibios.models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='sequencing',
            name='sample',
            field=models.ForeignKey(blank=True, null=True, on_delete=models.deletion.CASCADE, to='mibios_seq.Sample'),
        ),
    ]
