import unittest
from unittest.mock import patch
from main import handle_args


class TestHandleArgs(unittest.TestCase):

    # Test azure arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'azure'])
    def test_azure_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'azure')

    # Test openai arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'openai'])
    def test_openai_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'openai')

    @patch(
        'sys.argv',
        [
            'program',
            'input_file.epub',
            'output_folder',
            '--tts',
            'openai',
            '--normalize',
            '--package_m4b',
            '--openai_enable_polling',
            '--openai_submit_url',
            'https://example.com/jobs',
            '--openai_status_url_template',
            'https://example.com/jobs/{job_id}',
        ],
    )
    def test_optional_mvp_flags(self):
        config = handle_args()
        self.assertTrue(config.normalize)
        self.assertTrue(config.package_m4b)
        self.assertTrue(config.openai_enable_polling)

    # Test unsupported TTS provider
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'unsupported_tts'])
    def test_unsupported_tts(self):
        with self.assertRaises(SystemExit):  # argparse exits with SystemExit on error
            handle_args()

    # Test missing required input_file argument
    @patch('sys.argv', ['program', 'output_folder', '--tts', 'azure'])
    def test_missing_input_file(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test invalid log level argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--log', 'INVALID_LOG_LEVEL'])
    def test_invalid_log_level(self):
        with self.assertRaises(SystemExit):
            handle_args()


if __name__ == '__main__':
    unittest.main()
