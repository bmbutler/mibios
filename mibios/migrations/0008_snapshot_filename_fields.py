from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Migrate away from using FilePathField to store location for snapshot files

    The new scheme allows the database to relocate more easily as it stores
    just the file names and retrieves the directory from settings at
    runtime.FilePathField with absolute paths prevent.  The previous scheme
    also triggered the need to migrate when SNAPHOT_DIR changed.
    """

    dependencies = [
        ('mibios', '0007_add_snapshot_app_list'),
    ]

    def forward(apps, schema_editor):
        """
        Remove absolute paths prefixes

        Working on state with CharFields
        """
        Snapshot = apps.get_model('mibios', 'Snapshot')
        qs = Snapshot.objects.all()
        for i in qs:
            i.dbfile = i.dbfile.split('/')[-1]  # keep file name
            i.jsondump = i.jsondump.split('/')[-1]  # keep file name
        Snapshot.objects.bulk_update(qs, ['dbfile', 'jsondump'])

    def reverse(apps, schema_editor):
        """
        Add absolute snapshot dir as path prefixes

        Working on state with CharFields
        """
        Snapshot = apps.get_model('mibios', 'Snapshot')
        qs = Snapshot.objects.all()
        for i in qs:
            i.dbfile = settings.SNAPSHOT_DIR / i.dbfile
            i.jsondump = settings.SNAPSHOT_DIR / i.jsondump
        Snapshot.objects.bulk_update(qs, ['dbfile', 'jsondump'])

    operations = [
        migrations.AlterField(
            model_name='snapshot',
            name='dbfile',
            field=models.CharField(editable=False, max_length=500, verbose_name='archived database file'),
        ),
        migrations.AlterField(
            model_name='snapshot',
            name='jsondump',
            field=models.CharField(editable=False, max_length=500, verbose_name='JSON formatted archive'),
        ),
        migrations.RunPython(forward, reverse),
    ]
