

import os
__root_path__ = os.path.join(os.path.abspath(__file__).split('www_wiki')[0], 'www_wiki')
__raw_data_path__ = os.path.join(__root_path__, 'raw_data')
__save_data__ = os.path.join(__root_path__, 'processed')
__llm_result__ = os.path.join(__root_path__, 'llm_result')


