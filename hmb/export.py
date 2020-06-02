"""
Export data as files
"""
import sys

from .load import SequencingLoader
from .models import Sequencing


# TODO: add as method to model?
def to_meta_csv(seqs_qs=None, file=sys.stdout, sep='\t'):
    """
    Export sequencing records to meta format text file
    """
    if seqs_qs is None:
        seqs_qs = Sequencing.objects.all()

    head = [i for i, _ in SequencingLoader.COLS]
    file.write(sep.join(head) + '\n')
    rec = {}
    for i in seqs_qs:
        rec['fq_file_id'] = i.name
        rec['participant'] = i.sample.participant
        rec['sample_id'] = i.sample
        rec['week'] = i.sample.week
        rec['semester'] = i.sample.participant.semester

        rec['serial'] = i.batch.run.serial
        rec['run'] = i.batch.run.number
        rec['drop'] = '; '.join([str(j) for j in i.note.all()])

        row = []
        for col, name in SequencingLoader.COLS:
            row.append(str(rec.get(name, '')))

        file.write(sep.join(row) + '\n')
