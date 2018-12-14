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

/** Runs WordPiece tokenization.
  *
  * This tokenizer uses a greedy longest-match-first algorithm to perform tokenization using the provided vocabulary.
  *
  * For example:
  * {{{
  *   val input = "unaffable"
  *   val output = Seq("un", "##aff", "##able")
  * }}}
  *
  * @param  vocabulary    Vocabulary to use.
  * @param  unknownToken  Token used to represent unknown word pieces.
  * @param  maxWordLength Specifies the maximum allowed word length.
  *
  * @author Emmanouil Antonios Platanios
  */
class WordPieceTokenizer(
    override val vocabulary: Vocabulary,
    val unknownToken: String = "[UNK]",
    val maxWordLength: Int = 200
) extends Tokenizer {
  /** Tokenizes a piece of text. */
  override def tokenize(text: String): Seq[String] = {
    Tokenizer.splitOnWhitespace(text).flatMap(word => {
      if (word.length > maxWordLength) {
        Seq(unknownToken)
      } else {
        var isBad = false
        var start = 0
        var subTokens = Seq.empty[String]
        while (start < word.length) {
          // Find the longest matching substring.
          var end = word.length
          var currentSubstring = ""
          while (start < end) {
            var substring = word.slice(start, end)
            if (start > 0)
              substring = "##" + substring
            if (vocabulary.contains(substring)) {
              currentSubstring = substring
              start = end
            } else {
              end -= 1
            }
          }

          // Check if the substring is good.
          if (currentSubstring.isEmpty) {
            isBad = true
            start = word.length
          } else {
            subTokens :+= currentSubstring
            start = end
          }
        }
        if (isBad)
          Seq(unknownToken)
        else
          subTokens
      }
    })
  }
}
