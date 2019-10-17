import re
import unittest

from tf_core.nltoolkit.lib.textual_data_in_out import load_adc_from_tsv


class LoadDocumentCorpusFromTSV(unittest.TestCase):
    def test_load_adc_from_tsv(self):
        input_data = [('doc1', 'This is some text', '1'), ('doc2', 'This is some other text', '0')]
        header = ['title', 'text', 'label']

        input_text = []
        input_text.append('\t'.join(header))
        for line in input_data:
            input_text.append('\t'.join(line))

        input_dict = {
            'input': '\n'.join(input_text),
            'title_column': 'title',
            'text_column': 'text',
            'label_column': 'label'
        }
        output = load_adc_from_tsv(input_dict)
        adc = output['adc']

        self.assertEqual(len(input_data), len(adc.documents))
        for i, _ in enumerate(input_data):
            self.assertEqual(input_data[i][0], adc.documents[i].name)
            self.assertEqual(input_data[i][1], adc.documents[i].text)
            self.assertEqual(input_data[i][2], adc.documents[i].get_first_label())

    def test_custom_property_names(self):
        input_data = [('doc1', 'This is some text', '1'), ('doc2', 'This is some other text', '0')]
        header = ['some special title', 'some special text', 'some special label']

        input_text = []
        input_text.append('\t'.join(header))
        for line in input_data:
            input_text.append('\t'.join(line))

        input_dict = {
            'input': '\n'.join(input_text),
            'title_column': 'some special title',
            'text_column': 'some special text',
            'label_column': 'some special label'
        }
        output = load_adc_from_tsv(input_dict)
        adc = output['adc']

        self.assertEqual(len(input_data), len(adc.documents))
        for i, _ in enumerate(input_data):
            self.assertEqual(input_data[i][0], adc.documents[i].name)
            self.assertEqual(input_data[i][1], adc.documents[i].text)
            self.assertEqual(input_data[i][2], adc.documents[i].get_first_label())

    def test_multiple_spaces_in_text(self):
        input_data = [('doc1', 'This is                      some text', '1'),
                      ('doc2', 'This is some other text    ', '0')]
        header = ['title', 'text', 'label']

        input_text = []
        input_text.append('\t'.join(header))
        for line in input_data:
            input_text.append('\t'.join(line))

        input_dict = {
            'input': '\n'.join(input_text),
        }
        output = load_adc_from_tsv(input_dict)
        adc = output['adc']

        self.assertEqual(len(input_data), len(adc.documents))
        for i, _ in enumerate(input_data):
            self.assertEqual(input_data[i][0], adc.documents[i].name)
            # text gets normalized
            self.assertEqual(re.sub(' +', ' ', input_data[i][1]), adc.documents[i].text)
            self.assertEqual(input_data[i][2], adc.documents[i].get_first_label())

    def test_no_title(self):
        input_data = [('This is some text', '1'), ('This is some other text', '0')]
        header = ['text', 'label']

        input_text = []
        input_text.append('\t'.join(header))
        for line in input_data:
            input_text.append('\t'.join(line))

        input_dict = {
            'input': '\n'.join(input_text),
        }
        output = load_adc_from_tsv(input_dict)
        adc = output['adc']

        self.assertEqual(len(input_data), len(adc.documents))
        for i, _ in enumerate(input_data):
            self.assertEqual('', adc.documents[i].name)
            self.assertEqual(input_data[i][0], adc.documents[i].text)
            self.assertEqual(input_data[i][1], adc.documents[i].get_first_label())

    def test_no_label(self):
        input_data = [('doc1', 'This is some text'), ('doc2', 'This is some other text')]
        header = ['title', 'text']

        input_text = []
        input_text.append('\t'.join(header))
        for line in input_data:
            input_text.append('\t'.join(line))

        input_dict = {
            'input': '\n'.join(input_text),
        }
        output = load_adc_from_tsv(input_dict)
        adc = output['adc']

        self.assertEqual(len(input_data), len(adc.documents))
        for i, _ in enumerate(input_data):
            self.assertEqual(input_data[i][0], adc.documents[i].name)
            self.assertEqual(input_data[i][1], adc.documents[i].text)
            self.assertEqual('', adc.documents[i].get_first_label())

    def test_no_text(self):
        input_dict = {
            'input': 'some strange format',
        }
        self.assertRaises(Exception, load_adc_from_tsv, input_dict)


if __name__ == '__main__':
    unittest.main()
