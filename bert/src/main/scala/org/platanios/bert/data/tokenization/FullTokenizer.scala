/* Copyright 2018, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.bert.data.tokenization

/** Runs end-to-end tokenization including basic tokenization and WordPiece tokenization.
  *
  * @param  vocabulary    Vocabulary to use.
  * @param  caseSensitive Specifies whether or not to ignore case.
  *
  * @author Emmanouil Antonios Platanios
  */
class FullTokenizer(
    override val vocabulary: Vocabulary,
    val caseSensitive: Boolean = false,
    val unknownToken: String = "[UNK]",
    val maxWordLength: Int = 200
) extends Tokenizer {
  protected val basicTokenizer    : BasicTokenizer     = new BasicTokenizer(vocabulary, caseSensitive)
  protected val wordPieceTokenizer: WordPieceTokenizer = new WordPieceTokenizer(vocabulary, unknownToken, maxWordLength)

  /** Tokenizes a piece of text. */
  override def tokenize(text: String): Seq[String] = {
    basicTokenizer.tokenize(text).flatMap(wordPieceTokenizer.tokenize)
  }
}
