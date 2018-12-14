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

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
trait Tokenizer {
  val vocabulary: Vocabulary

  /** Tokenizes a piece of text. */
  def tokenize(text: String): Seq[String]

  def convertTokensToIDs(tokens: Seq[String]): Seq[Long] = {
    tokens.map(vocabulary.tokenToID)
  }

  def convertIDsToTokens(ids: Seq[Long]): Seq[String] = {
    ids.map(vocabulary.idToToken)
  }
}

object Tokenizer {
  val whitespaceRegex: Regex = "\\s+".r

  /** Splits the provided text on whitespace characters, after first removing any leading and trailing whitespace. */
  def splitOnWhitespace(text: String): Seq[String] = {
    whitespaceRegex.split(text.trim)
  }
}
