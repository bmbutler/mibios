from django.db import migrations


class Migration(migrations.Migration):
    """
    Data migration from old to new note model
    """

    dependencies = [
        ('mibios_seq', '0002_add_new_sequencing_note'),
    ]

    def forward(apps, schema_editor):
        HHCD_Note = apps.get_model('hhcd', 'Note')
        SeqNote = apps.get_model('mibios_seq', 'SeqNote')
        for i in HHCD_Note.objects.exclude(sequencing=None):
            if i.participant_set.exists():
                raise RuntimeError('unexpectedly related to participant')
            note = SeqNote(name=i.name, text=i.text)
            note.save()
            note.sequencing_set.set(i.sequencing_set.all())
            note.history.set(i.history.all())
            i.delete()

    def reverse(apps, schema_editor):
        HHCD_Note = apps.get_model('hhcd', 'Note')
        SeqNote = apps.get_model('mibios_seq', 'SeqNote')
        for i in SeqNote.objects.exclude(sequencing=None):
            note = HHCD_Note(name=i.name, text=i.text)
            note.save()
            note.sequencing_set.set(i.sequencing_set.all())
            note.history.set(i.history.all())
            i.delete()

    operations = [migrations.RunPython(forward, reverse),]
