from collections import Counter

from django.db import transaction

from hmb import models as m


class SequencingLoader():
    """
    Loader for data from sequencing tables

    These are tables that have one row per sequence data set (sample)
    """

    COLS = (
        ('FASTQ_ID', 'fq_file_id'),
        ('Participant_ID', 'participant'),
        ('Sample_ID', 'sample_id'),
        ('Study_week', 'week'),
        ('Semester', 'semester'),
        ('Use_Data', 'can_use'),
        ('Quantity_compliant', 'qcomp'),
        ('Frequency', 'freq'),
        ('Total_dose_grams', 'dose'),
        ('Supplement_consumed', 'supp'),
        ('pH', 'ph'),
        ('Bristol', 'bristol'),
        ('seq_serial', 'serial'),
        ('seq_run', 'run'),
        ('drop', 'drop'),
    )

    def __init__(self, *columns, sep='\t', add_only=True):
        valid_colnames = [i for i, j in self.COLS]
        for i in columns:
            if i not in valid_colnames:
                raise ValueError('Unknown column name: {}'.format(i))
        self.cols = columns
        self.sep = sep
        self.new_counter = Counter()
        self.add_only = add_only  # TODO: needs implementation
        for col, name in self.COLS:
            setattr(self, 'name', None)

    @classmethod
    @transaction.atomic
    def load_file(cls, file, sep='\t', add_only=True):
        colnames = file.readline().strip().split(sep)
        loader = cls(colnames, sep=sep, add_only=add_only)
        for i in file:
            loader.process_line(i)
        return loader.new_counter()

    def process_line(self, line):
        if type(line) == str:
            row = line.strip().split(self.sep)

        row = {k: v.strip() for k, v in zip(self.cols, row)}

        if 'participant' in row:
            participant, new = m.Participant.objects.get_or_create(
                name=row['participant']
            )
            if new:
                self.counter['participants'] += 1

        if 'week' in row:
            week, new = m.Week.objects.get_or_create(
                **m.Week.parse(row['week'])
            )
            if new:
                self.counter['weeks'] += 1

        if 'sample_id' in row:
            sample_id = m.FecesSample.parse_id(row['sample_id'])
            if 'participant' in row:
                if sample_id['participant'] != participant.name:
                    raise ValueError(
                        'Participant ID and SampleID inconsistent: {}'
                        ''.format(row)
                    )
                sample_id['participant'] = participant
            fecal_sample, new = m.FecesSample.objects.get_or_create(
                **sample_id,
            )
            if new:
                fecal_sample.week = week
                fecal_sample.save()
                self.counter['fecal samples'] += 1

        semester, new = m.Semester.objects.get_or_create(
            **m.Semester.parse(row['semester'])
        )
        if new:
            self.counter['semesters'] += 1

        run, new = m.SequencingRun.objects.get_or_create(
            serial=row['serial'], number=row['run']
        )
        if new:
            self.counter['runs'] += 1

        # FIXME: overwrites plate-level batches
        batch, new = m.SequencingBatch.objects.get_or_create(
            name='',  # empty, no plate information, whole run is batch
            run=run,
        )
        if new:
            self.counter['batches'] += 1

        # TODO: modify existing obj
        obj = m.Sequencing.objects.create(
            name=row['fq_file_id'],
            sample=fecal_sample,
            batch=batch,
        )
        self.counter['seq samples'] += 1

        if 'drop' in row:
            note, new = m.Note.objects.get_or_create(name=row['drop'])
            obj.note.add(note)
            if new:
                self.counter['notes'] += 1
