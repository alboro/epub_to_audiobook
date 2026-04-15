import unittest
from unittest.mock import patch
from main import handle_args


class TestHandleArgs(unittest.TestCase):

    # Test azure arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'azure'])
    def test_azure_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'azure')

    # Test openai arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'openai'])
    def test_openai_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'openai')

    @patch(
        'sys.argv',
        [
            'program',
            'input_file.epub',
            'output_folder',
            '--mode', 'all',
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

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'prepare'])
    def test_mode_prepare(self):
        config = handle_args()
        self.assertEqual(config.mode, 'prepare')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'audio', '--tts', 'edge'])
    def test_mode_audio(self):
        config = handle_args()
        self.assertEqual(config.mode, 'audio')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'package', '--tts', 'edge'])
    def test_mode_package(self):
        config = handle_args()
        self.assertEqual(config.mode, 'package')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'edge'])
    def test_mode_all(self):
        config = handle_args()
        self.assertEqual(config.mode, 'all')

    @patch('sys.argv', ['program', '/some/path/MyBook.epub', '--mode', 'prepare'])
    def test_default_output_folder(self):
        config = handle_args()
        self.assertIsNotNone(config.output_folder)
        self.assertTrue(config.output_folder.endswith('MyBook'))

    # Test unsupported TTS provider
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'unsupported_tts'])
    def test_unsupported_tts(self):
        with self.assertRaises(SystemExit):  # argparse exits with SystemExit on error
            handle_args()

    # Test missing required --mode argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'azure'])
    def test_missing_mode(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test missing required input_file argument
    @patch('sys.argv', ['program', '--mode', 'all', '--tts', 'azure'])
    def test_missing_input_file(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test invalid log level argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--log', 'INVALID_LOG_LEVEL'])
    def test_invalid_log_level(self):
        with self.assertRaises(SystemExit):
            handle_args()


if __name__ == '__main__':
    unittest.main()
