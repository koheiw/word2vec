/**
 * @file
 * @brief
 * @author Max Fomichev
 * @date 15.02.2017
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/
#include <Rcpp.h>
#include "word2vec.hpp"
#include "wordReader.hpp"
#include "vocabulary.hpp"
#include "trainer.hpp"

namespace w2v {
    bool w2vModel_t::train(const trainSettings_t &_trainSettings,
                           Texts texts,
                           vocabularyProgressCallback_t _vocabularyProgressCallback,
                           vocabularyStatsCallback_t _vocabularyStatsCallback,
                           trainProgressCallback_t _trainProgressCallback) noexcept {
        try {
            // map train data set file to memory
            // std::shared_ptr<fileMapper_t> trainWordsMapper(new fileMapper_t(_trainFile));
            // map stop-words file to memory
            // std::shared_ptr<fileMapper_t> stopWordsMapper;
            // if (!_stopWordsFile.empty()) {
            //     stopWordsMapper.reset(new fileMapper_t(_stopWordsFile));
            // }

            // build vocabulary, skip stop-words and words with frequency < minWordFreq
            std::shared_ptr<vocabulary_t> vocabulary(new vocabulary_t(texts,
                                                                      _trainSettings.minWordFreq,
                                                                      _vocabularyProgressCallback,
                                                                      _vocabularyStatsCallback));
            // key words descending ordered by their indexes
            std::vector<std::string> words;
            vocabulary->words(words);
            m_vectorSize = _trainSettings.size;
            m_mapSize = vocabulary->size();

            // train model
            std::vector<float> _trainMatrix;
            trainer_t(std::make_shared<trainSettings_t>(_trainSettings),
                      vocabulary,
                      texts,
                      _trainProgressCallback)(_trainMatrix);

            std::size_t wordIndex = 0;
            for (auto const &i:words) {
                auto &v = m_map[i];
                v.resize(m_vectorSize);
                std::copy(&_trainMatrix[wordIndex * m_vectorSize],
                          &_trainMatrix[(wordIndex + 1) * m_vectorSize],
                          &v[0]);
                wordIndex++;
            }

            return true;
        } catch (const std::exception &_e) {
            m_errMsg = _e.what();
        } catch (...) {
            m_errMsg = "unknown error";
        }

        return false;
    }
}
